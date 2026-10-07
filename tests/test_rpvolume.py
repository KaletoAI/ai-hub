"""Pin billed-volume ownership and restart state at the REST boundary."""
import asyncio
import copy
import json
import unittest

import httpx

import rpvolume


class FakeRest:
    def __init__(self):
        self.vols = []
        self.events = []
        self.drop_post = False
        self.wrapped = False

    def __call__(self, request):
        self.events.append(request.method)
        assert request.headers['authorization'] == 'Bearer api-key'
        assert request.extensions['timeout']['read'] == 30
        assert str(request.url).startswith('https://rest.runpod.io/v1/networkvolumes')
        if request.method == 'GET':
            return httpx.Response(200, json={'networkVolumes': self.vols} if self.wrapped else self.vols)
        if request.method == 'POST':
            body = json.loads(request.content)
            vol = dict(body, id='vol-1')
            self.vols.append(vol)
            if self.drop_post:
                self.drop_post = False
                raise httpx.ReadTimeout('lost answer', request=request)
            return httpx.Response(201, json=vol)
        vid = request.url.path.rsplit('/', 1)[1]
        vol = next(v for v in self.vols if v['id'] == vid)
        if request.method == 'PATCH':
            vol.update(json.loads(request.content))
            return httpx.Response(200, json=vol)
        self.vols.remove(vol)
        return httpx.Response(204)


class Lifecycle(unittest.TestCase):
    """Wrong ownership or retry decisions silently bill extra GB or delete models."""
    def setUp(self):
        self.rest = FakeRest()
        self.saved = None
        self.refs = []
        self.faults = []
        self.keys = ('api-key', 'access', 'secret')
        self.deps = rpvolume.VolumeDeps(
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(self.rest)),
            load_state=lambda name: self.saved,
            save_state=self.save, creds=lambda: self.keys,
            backends=lambda name: self.refs, alias_needs=lambda bid: [],
            alias_signature=lambda bid: '', source_index=lambda: {},
            url_catalog=lambda src: {}, lan=None, run_fetch=None,
            fetch_status=None, fetch_cancel=None,
            note_fault=lambda kind, detail: self.faults.append((kind, detail)), log=lambda text: None)
        self.cfg = {'datacenter': 'EU-RO-1', 'size_gb': 10, 'max_size_gb': 40}
        self.ctl = rpvolume.VolumeController('models', self.cfg, self.deps)

    def save(self, name, state):
        self.saved = copy.deepcopy(state)
        self.rest.events.append('save')

    def ensure(self, create=True):
        return asyncio.run(self.ctl.ensure_volume(create))

    def test_create_timeout_adopts(self):
        """A lost POST answer is adopted by name, never billed a second time."""
        self.rest.drop_post = True
        self.assertIsNone(self.ensure())
        self.assertEqual(self.ensure()['id'], 'vol-1')
        self.assertEqual(self.rest.events.count('POST'), 1)
        self.assertEqual(self.saved['id'], 'vol-1')

    def test_adopt_requires_dc_match(self):
        """A same-named volume in another DC must not be adopted or duplicated."""
        self.rest.vols = [dict(id='other', name='aihub-models', dataCenterId='US-GA-2', size=10)]
        self.assertIsNone(self.ensure())
        self.assertTrue(any('DC' in p for p in self.ctl.problems()))
        self.assertNotIn('POST', self.rest.events)

    def test_gone_only_after_two_lists(self):
        """Transient absence retains identity; confirmed loss requires explicit creation."""
        self.ensure()
        self.rest.vols.clear()
        self.assertIsNone(self.ensure(False))
        self.assertEqual(self.saved['missing'], 1)
        self.assertFalse(any('deleted outside' in p for p in self.ctl.problems()))
        self.assertIsNone(self.ensure(False))
        self.assertEqual(self.saved['missing'], 2)
        self.assertTrue(any('deleted outside' in p for p in self.ctl.problems()))
        self.assertEqual(self.rest.events.count('POST'), 1)
        self.assertIsNotNone(self.ensure(True))
        self.assertEqual(self.saved['missing'], 0)

    def test_grow_only_on_fresh_list_and_never_shrinks(self):
        """Growth must use current ownership and size, never a stale id."""
        self.ensure()
        self.assertTrue(asyncio.run(self.ctl.grow_to(20)))
        self.assertEqual(self.rest.events[-4:], ['GET', 'save', 'PATCH', 'save'])
        self.assertEqual(self.saved['size_gb'], 20)
        self.assertIn('0.70', self.faults[0][1])
        self.assertIn('1.40', self.faults[0][1])
        self.assertTrue(asyncio.run(self.ctl.grow_to(10)))
        self.rest.vols.clear()
        self.assertFalse(asyncio.run(self.ctl.grow_to(30)))
        self.assertEqual(self.rest.events.count('PATCH'), 1)

    def test_partial_rest_answers_still_saved(self):
        """The PATCH/POST answer shapes are unverified [U]: an answer without
        dataCenterId or size must still leave the id and the new size saved — a KeyError
        after the billed call would lose them and the next round would grow again."""
        orig = self.rest.__call__

        def partial(request):
            r = orig(request)
            if request.method == 'PATCH':
                return httpx.Response(200, json={})
            if request.method == 'POST':
                return httpx.Response(201, json={'id': 'vol-1'})
            return r
        self.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(partial))
        self.ensure()
        self.assertEqual((self.saved['id'], self.saved['size_gb'], self.saved['dc']),
                         ('vol-1', 10, 'EU-RO-1'))
        self.assertTrue(asyncio.run(self.ctl.grow_to(20)))
        self.assertEqual(self.saved['size_gb'], 20)

    def test_delete_refused_with_referencing_backend(self):
        """A referenced volume cannot disappear beneath its backend."""
        self.ensure()
        self.refs = [{'id': 'b'}]
        self.assertTrue(asyncio.run(self.ctl.delete_volume()))
        self.assertNotIn('DELETE', self.rest.events)

    def test_state_saved_before_next_await(self):
        """New identity reaches persistence before even client cleanup can yield."""
        rest = self.rest
        owner = self
        class Client(httpx.AsyncClient):
            async def __aexit__(self, *args):
                owner.assertEqual(owner.saved['id'], 'vol-1')
                rest.events.append('close')
                await super().__aexit__(*args)
        self.deps.client_factory = lambda: Client(transport=httpx.MockTransport(rest))
        self.ensure()
        self.assertEqual(rest.events, ['GET', 'POST', 'save', 'close'])

    def test_mutations_refuse_changed_identity(self):
        """An id with a changed name or DC cannot authorize money mutations."""
        for field, value in [('name', 'foreign'), ('dataCenterId', 'US-GA-2')]:
            with self.subTest(field=field):
                self.rest.vols = [dict(id='vol-1', name='aihub-models', dataCenterId='EU-RO-1', size=10)]
                self.ensure()
                self.rest.vols[0][field] = value
                self.assertFalse(asyncio.run(self.ctl.grow_to(20)))
                self.assertTrue(asyncio.run(self.ctl.delete_volume()))
        self.assertNotIn('PATCH', self.rest.events)
        self.assertNotIn('DELETE', self.rest.events)

    def test_delete_fresh_and_clears_saved_state(self):
        """Deletion verifies ownership afresh and removes stale transfer identity."""
        self.ensure()
        self.ctl.state['fetch_job'] = {'id': 'job'}
        self.assertEqual(asyncio.run(self.ctl.delete_volume()), '')
        self.assertEqual(self.rest.events[-4:], ['GET', 'save', 'DELETE', 'save'])
        self.assertIsNone(self.saved['id'])
        self.assertIsNone(self.saved['fetch_job'])

    def test_state_defaults_and_corrupt_state(self):
        """Unreadable persisted identity must never become a fresh billed volume."""
        self.saved = {'id': 'old', 'mpu': [{'key': 'x', 'upload_id': 'u'}]}
        ctl = rpvolume.VolumeController('models', self.cfg, self.deps)
        self.assertEqual(ctl.state['mpu'], self.saved['mpu'])
        self.assertEqual(ctl.state['blocked'], {})
        self.assertEqual(ctl.state['dc'], 'EU-RO-1')
        self.saved = []
        with self.assertRaises(ValueError):
            rpvolume.VolumeController('models', self.cfg, self.deps)

    def test_missing_credentials_checklist(self):
        """Missing secrets are actionable without issuing unauthenticated creates."""
        self.keys = ('', '', '')
        self.assertIsNone(self.ensure())
        self.assertEqual(len(self.ctl.problems()), 2)
        self.assertEqual(self.rest.events, [])


    def test_grow_ceiling(self):
        """A direct lifecycle call cannot spend beyond the configured ceiling."""
        self.ensure()
        self.assertFalse(asyncio.run(self.ctl.grow_to(41)))
        self.assertNotIn('PATCH', self.rest.events)
        self.assertEqual(self.saved['size_gb'], 10)

    def test_delete_absent_id(self):
        """A persisted id alone cannot authorize deleting a missing volume."""
        self.ensure()
        self.rest.vols.clear()
        self.assertTrue(asyncio.run(self.ctl.delete_volume()))
        self.assertNotIn('DELETE', self.rest.events)
        self.assertEqual(self.saved['id'], 'vol-1')

    def test_rejected_key_checklist_changes_with_secret(self):
        """A rejected API key remains visible until the operator replaces it."""
        self.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(403, text='denied')))
        with self.assertRaises(rpvolume.RestAuthError):
            self.ensure()
        self.assertIn('API key rejected', self.ctl.problems())
        self.keys = ('replacement', 'access', 'secret')
        self.assertNotIn('API key rejected', self.ctl.problems())

    def test_malformed_list_never_creates(self):
        """An unexpected list schema must not masquerade as an empty account."""
        events = []
        def malformed(request):
            events.append(request.method)
            return httpx.Response(200, json={'unexpected': []})
        self.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(malformed))
        with self.assertRaises(rpvolume.RestError):
            self.ensure()
        self.assertEqual(events, ['GET'])


class Rest(unittest.TestCase):
    """REST shape and status mistakes can make a correct volume look absent."""
    def test_list_shapes_and_payloads(self):
        """Both list schemas retain records and create uses the provider field names."""
        async def check():
            fake = FakeRest()
            async with httpx.AsyncClient(transport=httpx.MockTransport(fake)) as client:
                await rpvolume.create_volume(client, 'api-key', 'models', 10, 'EU-RO-1')
                for wrapped in (False, True):
                    fake.wrapped = wrapped
                    self.assertEqual(await rpvolume.list_volumes(client, 'api-key'),
                                     [dict(id='vol-1', name='aihub-models', size=10, dataCenterId='EU-RO-1')])
        asyncio.run(check())

    def test_errors(self):
        """Authentication errors stay distinguishable and response bodies are bounded."""
        async def check():
            for status in (401, 403, 429, 500):
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                        lambda request: httpx.Response(status, text='x' * 250))) as client:
                    cls = rpvolume.RestAuthError if status in (401, 403) else rpvolume.RestError
                    with self.assertRaises(cls) as caught:
                        await rpvolume.list_volumes(client, 'api-key')
                    self.assertEqual(caught.exception.status, status)
                    self.assertIn('x' * 200, str(caught.exception))
                    self.assertNotIn('x' * 201, str(caught.exception))
        asyncio.run(check())


from tests.test_s3vol import FakeS3
from tests.test_modelsync import need
import s3vol


class SizedBody:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class PlanRound(unittest.TestCase):
    """S3 planning must not route missing weights or delete unowned objects."""
    def setUp(self):
        self.fake = FakeS3()
        self.rest = FakeRest()
        self.rest.vols = [dict(id='vol-1', name='aihub-models', dataCenterId='EU-RO-1', size=10)]
        self.src = {'models/a.safetensors': 5}
        self.needs = [need('A', [], catalog=['models/a.safetensors'])]
        self.refs = [{'name': 'worker'}]
        self.logs, self.saved, self.bids = [], [], []
        self.urls = {}
        self.keys = ('api-key', 'access', 'secret')
        def alias_needs(bid):
            self.bids.append(bid)
            return self.needs
        def transport(request):
            return self.rest(request) if request.url.host == 'rest.runpod.io' else self.fake.handler(request)
        deps = rpvolume.VolumeDeps(
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport)),
            load_state=lambda name: {'id': 'vol-1', 'size_gb': 10},
            save_state=lambda name, state: self.saved.append(copy.deepcopy(state)),
            creds=lambda: self.keys, backends=lambda name: self.refs,
            alias_needs=alias_needs, alias_signature=lambda bid: '',
            source_index=lambda: self.src, url_catalog=lambda src: self.urls,
            lan=None, run_fetch=None, fetch_status=None, fetch_cancel=None,
            note_fault=lambda kind, detail: None, log=self.logs.append,
            s3_factory=lambda client: s3vol.S3Volume(client, s3vol.endpoint_for('EU-RO-1'),
                                                  'vol-1', 'access', 'secret', 'eu-ro-1'))
        self.ctl = rpvolume.VolumeController('models',
            {'datacenter': 'EU-RO-1', 'size_gb': 10, 'max_size_gb': 40}, deps)

    def run_round(self):
        asyncio.run(self.ctl.plan_round())

    def manifest(self, data):
        self.fake.objs['.gw-modelsync.json'] = json.dumps(data).encode()

    def test_plan_reads_volume_and_manifest(self):
        """Both S3 prefixes and verified manifest sizes must contribute to readiness."""
        self.fake.objs.update({'models/a.safetensors': b'12345', 'hf-cache/b': b'12'})
        self.needs = [need('A', [], catalog=['models/a.safetensors', 'hf-cache/b'])]
        self.manifest({'hf-cache/b': {'size': 2, 'aliases': ['A']}})
        self.run_round()
        self.assertTrue(self.ctl.is_alias_ready('A'))
        self.assertIn('ready', self.ctl.alias_status('A'))
        self.assertEqual(self.ctl.view()['used_bytes'], 7)
        self.assertEqual(self.bids, ['runpod:worker'])

    def test_gw_part_is_not_present(self):
        """A worker partial must never open the routing gate."""
        self.fake.objs['models/a.safetensors.gw-part'] = b'12345'
        self.run_round()
        self.assertFalse(self.ctl.is_alias_ready('A'))
        self.assertNotIn('models/a.safetensors.gw-part', self.ctl.dest)

    def test_grow_when_short(self):
        """Headroom growth must include unknown occupied bytes and missing weights."""
        self.fake.objs['models/unknown'] = SizedBody(6_000_000_000)
        self.src['models/a.safetensors'] = 5_000_000_000
        self.run_round()
        self.assertEqual(self.rest.vols[0]['size'], 13)
        self.assertEqual(self.ctl.state['size_gb'], 13)

    def test_grow_capped(self):
        """A bad alias cannot order capacity beyond the operator's ceiling."""
        self.ctl.cfg['max_size_gb'] = 12
        self.src['models/a.safetensors'] = 20_000_000_000
        self.run_round()
        self.assertLessEqual(self.rest.vols[0]['size'], 12)
        self.assertIn('limit 12 GB', self.ctl.alias_status('A'))
        self.assertFalse(self.ctl.is_alias_ready('A'))

    def test_prune_only_under_pressure(self):
        """Owned leftovers stay until pressure, then deletion also updates the manifest."""
        self.fake.objs['models/old'] = SizedBody(6_000_000_000)
        self.manifest({'models/old': {'size': 6_000_000_000}})
        self.run_round()
        self.assertIn('models/old', self.fake.objs)
        self.src['models/a.safetensors'] = 5_000_000_000
        self.run_round()
        self.assertNotIn('models/old', self.fake.objs)
        self.assertEqual(json.loads(self.fake.objs['.gw-modelsync.json']), {})
        self.assertNotIn('PATCH', self.rest.events)

    def test_unknown_never_deleted_automatically(self):
        """Space pressure cannot authorize deleting a file outside the manifest."""
        self.fake.objs['models/unknown'] = SizedBody(6_000_000_000)
        self.src['models/a.safetensors'] = 5_000_000_000
        self.run_round()
        self.assertIn(['models/unknown', 6_000_000_000], self.ctl.plan['unknown'])
        self.assertIn('models/unknown', self.fake.objs)

    def test_status_texts_no_io(self):
        """Request-time readiness and console views must use the cached plan."""
        self.run_round()
        def forbidden(request):
            raise AssertionError('request-time I/O')
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
        self.assertFalse(self.ctl.is_alias_ready('A'))
        self.assertIn('syncing', self.ctl.alias_status('A'))
        view = self.ctl.view()
        self.assertEqual(view['phase'], 'syncing')
        self.assertEqual(view['backends'], ['worker'])
        self.assertAlmostEqual(view['cost_month_usd'], .70)

    def test_unreadable_manifest_is_empty_not_fatal(self):
        """Corrupt ownership metadata must leave files unknown and undeleted."""
        self.fake.objs.update({'models/old': b'123', '.gw-modelsync.json': b'bad json'})
        self.run_round()
        self.assertEqual(self.ctl.manifest, {})
        self.assertIn(['models/old', 3], self.ctl.plan['unknown'])
        self.assertTrue(self.logs)
        self.assertIn('models/old', self.fake.objs)

    def test_idle_never_prunes(self):
        """Removing the final backend cannot trigger automatic cleanup."""
        self.refs = []
        self.fake.objs['models/old'] = SizedBody(11_000_000_000)
        self.manifest({'models/old': {'size': 11_000_000_000}})
        self.run_round()
        self.assertEqual(self.ctl.plan['prune'], [])
        self.assertIn('models/old', self.fake.objs)

    def test_s3_error_keeps_last_good_gate(self):
        """A 503 from the S3 API says nothing about the files the workers mount: the
        last good snapshot stays, or every S3 hiccup would 503 the RunPod aliases. The
        error is shown and kept for the loop's backoff."""
        self.fake.objs['models/a.safetensors'] = b'12345'
        self.run_round()
        self.assertTrue(self.ctl.is_alias_ready('A'))
        self.fake.handler = lambda request: httpx.Response(503)
        self.run_round()
        self.assertTrue(self.ctl.is_alias_ready('A'))
        self.assertTrue(self.ctl.view()['sync_error'])
        self.assertIsInstance(self.ctl.last_error, s3vol.S3Error)

    def test_gate_stays_open_during_a_round(self):
        """The snapshot is swapped at the END of a round: cleared at its start, every
        10-min round closed the gate for as long as its S3 listing took."""
        self.fake.objs['models/a.safetensors'] = b'12345'
        self.run_round()
        seen = []
        orig = self.fake.handler

        def spy(request):
            seen.append(self.ctl.is_alias_ready('A'))
            return orig(request)
        self.fake.handler = spy
        self.run_round()
        self.assertTrue(seen)
        self.assertTrue(all(seen))

    def test_gone_volume_closes_gate(self):
        """A volume missing from two lists has no models: its snapshot goes."""
        self.fake.objs['models/a.safetensors'] = b'12345'
        self.run_round()
        self.ctl.state['missing'] = rpvolume.GONE_AFTER
        self.run_round()
        self.assertFalse(self.ctl.is_alias_ready('A'))

    def test_missing_identity_or_secret_stops_round(self):
        """Missing credentials or identity must never issue an S3 request."""
        self.keys = ('api-key', '', '')
        self.run_round()
        self.assertTrue(self.ctl.problems())
        self.assertEqual(self.fake.calls, [])
        self.keys = ('api-key', 'access', 'secret')
        self.ctl.state['id'] = None
        self.run_round()
        self.assertTrue(self.ctl.problems())
        self.assertEqual(self.fake.calls, [])

    def test_manifest_link_overlay(self):
        """S3's symlink size must not hide a verified link or double-count its target."""
        self.src = {'hf-cache/b': 2}
        self.needs = [need('A', [], catalog=['hf-cache/a.safetensors'])]
        self.fake.objs.update({'hf-cache/a.safetensors': b'12', 'hf-cache/b': b'12'})
        self.manifest({'hf-cache/a.safetensors': {'source': 'link', 'target': 'b'},
                       'hf-cache/b': {'size': 2}})
        self.run_round()
        self.assertTrue(self.ctl.is_alias_ready('A'))
        self.assertEqual(self.ctl.view()['used_bytes'], 2)

    def test_stale_fallback_saved(self):
        """A changed catalog URL must release the old fallback and its stale reason."""
        self.ctl.state['url_fallback'] = {'models/a.safetensors': 'https://old'}
        self.ctl.state['url_fallback_why'] = {'models/a.safetensors': 'old failure'}
        self.urls = {'models/a.safetensors': {'url': 'https://new'}}
        self.run_round()
        self.assertEqual(self.ctl.plan['fetch'][0]['source'], 'url')
        self.assertEqual(self.saved[-1]['url_fallback'], {})
        self.assertEqual(self.saved[-1]['url_fallback_why'], {})

    def test_capacity_blocks_clear_after_limit_change(self):
        """Raising the ceiling must release a capacity block without clearing other failures."""
        self.ctl.cfg['max_size_gb'] = 12
        self.src['models/a.safetensors'] = 20_000_000_000
        self.run_round()
        self.ctl.state['blocked']['models/other'] = 'share file changing?'
        self.ctl.cfg['max_size_gb'] = 40
        self.run_round()
        self.assertNotIn('models/a.safetensors', self.ctl.state['blocked'])
        self.assertEqual(self.ctl.state['blocked']['models/other'], 'share file changing?')

    def test_view_fetch_job_omits_urls(self):
        """A console snapshot must never expose signed catalog URLs from job items."""
        self.ctl.state['fetch_job'] = {'id': 'job-1', 'bid': 'runpod:worker', 'ts': 1,
            'items': [{'path': 'models/a.safetensors', 'size': 5,
                       'url': 'https://example/model?token=private'}]}
        self.run_round()
        view = self.ctl.view()
        self.assertEqual(view['fetch_job']['id'], 'job-1')
        self.assertNotIn('private', json.dumps(view))

    def test_prune_requires_fresh_ownership(self):
        """A saved id with changed ownership must not authorize S3 deletion."""
        self.fake.objs['models/old'] = SizedBody(6_000_000_000)
        self.manifest({'models/old': {'size': 6_000_000_000}})
        self.src['models/a.safetensors'] = 5_000_000_000
        self.rest.vols[0]['name'] = 'foreign'
        self.run_round()
        self.assertIn('models/old', self.fake.objs)
        self.assertTrue(self.ctl.problems())

    def test_prune_skips_inflight_paths(self):
        """Space pressure cannot delete a path held by a saved fetch or multipart writer."""
        for field, value in [('fetch_job', {'id': 'job', 'items': [{'path': 'models/old'}]}),
                             ('mpu', [{'key': 'models/old', 'upload_id': 'upload'}])]:
            with self.subTest(field=field):
                self.fake.objs['models/old'] = SizedBody(6_000_000_000)
                self.manifest({'models/old': {'size': 6_000_000_000}})
                self.src['models/a.safetensors'] = 5_000_000_000
                self.ctl.state[field] = value
                self.run_round()
                self.assertIn('models/old', self.fake.objs)
                self.ctl.state[field] = None if field == 'fetch_job' else []
