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
        self.assertEqual(view['phase'], 'waiting')
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


import hashlib
import sys
from unittest.mock import patch


class TransferRound(unittest.TestCase):
    """Transfer tests exercise the controller against local bytes and fake providers."""
    def setUp(self):
        PlanRound.setUp(self)
        self.payloads, self.faults = [], []
        self.result_ok = True
        self.error = 'temporary failure'
        self.publish = True
        self.body = b'12345'
        outer = self
        class Lan:
            generation = 0
            def usable(self): return True
            def problem(self): return 'offline'
            def cat_argv(self, path, off):
                assert off == 0
                return [sys.executable, '-c', 'import sys; sys.stdout.buffer.write(' + repr(outer.body) + ')']
            async def sha256(self, path, size):
                return hashlib.sha256(outer.body).hexdigest()
        self.ctl.deps.lan = Lan()
        self.ctl.deps.note_fault = lambda kind, detail: self.faults.append((kind, detail))
        async def run_fetch(bid, payload, on_id, *, max_wait=None):
            self.payloads.append(payload)
            on_id('j1')
            self.assertEqual(self.saved[-1]['fetch_job']['id'], 'j1')
            results = []
            for item in payload.get('items', []):
                if self.publish and self.result_ok:
                    self.fake.objs[item['path']] = self.body
                results.append(dict(path=item['path'], ok=self.result_ok, size=5,
                                    sha256=hashlib.sha256(self.body).hexdigest(), error=self.error))
            return dict(status='COMPLETED', output={'results': results}, executionTime=3600000)
        self.ctl.deps.run_fetch = run_fetch
        self.ctl.deps.fetch_status = lambda bid, jid: run_fetch(bid, {'items': self.ctl.state['fetch_job']['items']}, lambda jid: None)
        async def cancel(bid, jid): return True
        self.ctl.deps.fetch_cancel = cancel
        self.refs[0]['cost_per_hour'] = 2

    run_round = PlanRound.run_round
    manifest = PlanRound.manifest

    async def transfer(self):
        await self.ctl.resume()
        await self.ctl.plan_round()
        # Older round tests exercise a whole sync; each production tick now does one unit.
        for _ in range(4):
            await self.ctl._transfer_round()
            if self.ctl.attempts or self.ctl._stream_failures or not self.ctl._resumed:
                break
            if self.ctl.plan is None or not self.ctl.plan['fetch'] or not self.ctl._can_progress():
                break

    def test_job_and_lan_split(self):
        """Catalog URLs must bypass the uplink while LAN-only files use S3."""
        self.src['models/b'] = 5
        self.needs = [need('A', [], catalog=list(self.src))]
        self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
        asyncio.run(self.transfer())
        self.assertEqual([i['path'] for i in self.payloads[0]['items']], ['models/a.safetensors'])
        self.assertEqual(self.fake.objs['models/b'], b'12345')
        self.assertEqual(self.ctl.manifest['models/b']['source'], 'lan')
        self.assertAlmostEqual(self.ctl.state['sync_cost_usd'], 2)

    def test_fetch_payload_has_no_secret(self):
        """Fetch input cannot carry gateway or S3 credentials to the worker."""
        self.keys = ('hf-private-token', 'access', 's3-private-secret')
        self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
        asyncio.run(self.transfer())
        self.assertNotIn('private', json.dumps(self.payloads))

    def test_job_id_saved_before_poll(self):
        """An id must survive a restart before run_fetch begins polling."""
        self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
        asyncio.run(self.transfer())
        saved = next(s['fetch_job'] for s in self.saved if s['fetch_job'])
        self.assertEqual(saved['bid'], 'runpod:worker')
        self.assertEqual(saved['items'], [{'path': 'models/a.safetensors', 'size': 5, 'url': 'https://example/a', 'aliases': ['A']}])

    def test_manifest_only_after_head(self):
        """A worker's ok flag alone cannot publish absent model ownership."""
        self.publish = False
        self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
        with self.assertRaises(RuntimeError): asyncio.run(self.transfer())
        self.assertNotIn('models/a.safetensors', self.ctl.manifest)
        self.assertEqual(self.ctl.attempts, {})

    def test_url_failure_falls_back_to_lan_then_blocks_without_share(self):
        """Final URL failures must select a share copy or block with a fault."""
        self.result_ok = False
        self.error = 'final: size mismatch'
        self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
        asyncio.run(self.transfer())
        self.assertEqual(self.ctl.state['url_fallback']['models/a.safetensors'], 'https://example/a')
        asyncio.run(self.ctl._transfer_round())
        self.assertEqual(self.ctl.manifest['models/a.safetensors']['source'], 'lan')
        self.setUp()
        self.src = {}
        self.manifest({'models/a.safetensors': {'size': 5}})
        self.result_ok = False
        self.error = 'final: HTTP 404'
        self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
        asyncio.run(self.transfer())
        self.assertIn('models/a.safetensors', self.ctl.state['blocked'])
        self.assertTrue(self.faults)

    def test_multipart_abort_on_sha_mismatch(self):
        """Changed share bytes must abort parts and block after two failed streams."""
        async def bad_hash(path, size): return 'wrong'
        self.ctl.deps.lan.sha256 = bad_hash
        with patch.object(rpvolume, 'PART_SIZE', 3, create=True):
            asyncio.run(self.transfer())
            asyncio.run(self.ctl._transfer_round())
        self.assertEqual(self.fake.mpu, {})
        self.assertNotIn('models/a.safetensors', self.ctl.manifest)
        self.assertEqual(self.ctl.state['blocked']['models/a.safetensors'], 'share file changing?')

    def test_complete_timeout_with_right_head_is_success(self):
        """A lost completion answer must accept verified bytes instead of duplicating them."""
        handler = self.fake.handler
        def timeout(req):
            response = handler(req)
            if req.method == 'POST' and 'uploadId' in req.url.params:
                raise httpx.ReadTimeout('lost answer')
            return response
        self.fake.handler = timeout
        with patch.object(rpvolume, 'PART_SIZE', 3, create=True):
            asyncio.run(self.transfer())
        self.assertEqual(self.ctl.manifest['models/a.safetensors']['size'], 5)
        self.assertEqual(self.ctl.state['mpu'], [])

    def test_never_two_writers(self):
        """A concurrent round cannot stream the path held by a fetch job."""
        async def check():
            started, release = asyncio.Event(), asyncio.Event()
            orig = self.ctl.deps.run_fetch
            async def pending(bid, payload, on_id, *, max_wait=None):
                on_id('pending'); started.set()
                await release.wait()
                return await orig(bid, payload, on_id)
            self.ctl.deps.run_fetch = pending
            self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
            await self.ctl.resume(); await self.ctl.plan_round()
            task = asyncio.create_task(self.ctl._transfer_round())
            await started.wait()
            self.ctl.urls = {}
            await self.ctl._transfer_round()
            self.assertNotIn('models/a.safetensors', self.fake.objs)
            release.set(); await task
        asyncio.run(check())

    def test_resume_finishes_saved_job_before_new_writer(self):
        """Restart must settle the saved writer before launching another one."""
        self.ctl.state['fetch_job'] = {'id': 'old', 'bid': 'runpod:worker', 'ts': 0,
            'items': [{'path': 'models/a.safetensors', 'size': 5, 'url': 'https://example/a'}]}
        events = []
        async def status(bid, jid):
            events.append((bid, jid))
            self.fake.objs['models/a.safetensors'] = b'12345'
            return {'status': 'COMPLETED', 'output': {'results': [{'path': 'models/a.safetensors', 'ok': True, 'sha256': 'abc'}]}}
        self.ctl.deps.fetch_status = status
        asyncio.run(self.transfer())
        self.assertEqual(events, [('runpod:worker', 'old')])
        self.assertEqual(self.payloads, [])
        self.assertEqual(self.ctl.state['fetch_job'], None)

    def test_resume_aborts_saved_and_listed_uploads(self):
        """Restart cannot leave persisted or unrecorded multipart storage orphaned."""
        self.fake.mpu = {'saved': ('models/a', {}), 'lost': ('hf-cache/b', {})}
        self.ctl.state['mpu'] = [{'key': 'models/a', 'upload_id': 'saved'}]
        asyncio.run(self.ctl.resume())
        self.assertEqual(self.fake.mpu, {})
        self.assertEqual(self.ctl.state['mpu'], [])

    def test_delete_unknown_only_unknown_and_not_inflight(self):
        """Operator cleanup cannot erase owned, arbitrary or actively written objects."""
        self.fake.objs.update({'models/unknown': b'1', 'models/held': b'2', 'models/owned': b'3'})
        self.manifest({'models/owned': {'size': 1}})
        self.run_round()
        self.ctl._inflight.add('models/held')
        count = asyncio.run(self.ctl.delete_unknown(['models/unknown', 'models/held', 'models/owned', '.gw-modelsync.json']))
        self.assertEqual(count, 1)
        self.assertIn('models/owned', self.fake.objs)
        self.assertIn('models/held', self.fake.objs)

    def test_backoff_doubles_and_resets(self):
        """Transient failures must delay retries and a good round resets the delay."""
        async def check():
            clock = [0]
            self.ctl.deps.now = lambda: clock[0]
            calls = []
            original = self.fake.handler
            def failing(req): return httpx.Response(503)
            self.fake.handler = failing
            async def tick(seconds):
                calls.append(clock[0]); clock[0] += seconds
                if clock[0] == 85: self.fake.handler = original
                if clock[0] >= 105: await self.ctl.aclose()
            with patch('rpvolume.asyncio.sleep', tick):
                await self.ctl.run_forever()
            self.assertEqual(self.ctl._backoff, 0)
            self.assertIn(30, calls)
            self.assertIn(90, calls)
        asyncio.run(check())

    def test_auth_error_pauses_until_creds_change(self):
        """Rejected credentials must pause S3 requests until the tuple changes."""
        async def check():
            clock = [0]; rejected_calls = []
            self.ctl.deps.now = lambda: clock[0]
            original = self.fake.handler
            def rejected(req):
                rejected_calls.append(clock[0]); return httpx.Response(403)
            self.fake.handler = rejected
            async def tick(seconds):
                clock[0] += seconds
                if clock[0] == 10:
                    self.assertIn('S3 key rejected', self.ctl.problems())
                if clock[0] == 15:
                    self.keys = ('api-key', 'new-access', 'new-secret'); self.fake.handler = original
                if clock[0] >= 25: await self.ctl.aclose()
            with patch('rpvolume.asyncio.sleep', tick): await self.ctl.run_forever()
            self.assertEqual(rejected_calls, [0])
            self.assertNotIn('S3 key rejected', self.ctl.problems())
        asyncio.run(check())

    def test_fetch_batches_obey_both_limits(self):
        """Batching must preserve plan order without exceeding either worker limit."""
        self.src = {'models/a': 5, 'models/b': 5, 'models/c': 5}
        self.needs = [need('A', [], catalog=list(self.src))]
        self.urls = {p: {'url': 'https://example/' + p} for p in self.src}
        with patch.object(rpvolume, 'JOB_BYTES', 10), patch.object(rpvolume, 'JOB_FILES', 1):
            asyncio.run(self.transfer())
        self.assertEqual([[i['path'] for i in p['items']] for p in self.payloads],
                         [['models/a'], ['models/b'], ['models/c']])

    def test_links_only_after_target_and_resume_link_job(self):
        """A link needs a present blob, and its job must remain recoverable across restart."""
        self.src = {'hf-cache/blobs/b': 5, 'hf-cache/snapshots/v/a': {'link': '../../blobs/b'}}
        self.needs = [need('A', [], catalog=['hf-cache/snapshots/v/a'])]
        orig = self.ctl.deps.run_fetch
        async def run(bid, payload, on_id, *, max_wait=None):
            if payload['op'] != 'link': return await orig(bid, payload, on_id)
            self.assertIn('hf-cache/blobs/b', self.fake.objs)
            on_id('link-job')
            return {'status': 'COMPLETED', 'output': {'results': [{'path': payload['links'][0]['path'], 'ok': True}]}}
        self.ctl.deps.run_fetch = run
        asyncio.run(self.transfer())
        self.assertEqual(self.ctl.manifest['hf-cache/snapshots/v/a']['target'], '../../blobs/b')
        self.assertTrue(self.ctl.is_alias_ready('A'))

    def test_resume_unconfirmed_cancel_keeps_gate_closed(self):
        """A deadline cannot authorize a second writer when cancel was not confirmed."""
        self.ctl.state['fetch_job'] = {'id': 'old', 'bid': 'runpod:worker', 'ts': 0,
            'items': [{'path': 'models/a.safetensors', 'size': 5, 'url': 'https://example/a'}]}
        async def check():
            clock = [0]
            self.ctl.deps.now = lambda: clock[0]
            async def status(bid, jid): return {'status': 'IN_PROGRESS'}
            async def cancel(bid, jid): return False
            async def tick(seconds): clock[0] += 1800
            self.ctl.deps.fetch_status, self.ctl.deps.fetch_cancel = status, cancel
            with patch('rpvolume.asyncio.sleep', tick): await self.ctl.resume()
            await self.ctl.plan_round(); await self.ctl._transfer_round()
            self.assertFalse(self.ctl._resumed)
            self.assertIsNotNone(self.ctl.state['fetch_job'])
            self.assertNotIn('models/a.safetensors', self.fake.objs)
        asyncio.run(check())

    def test_sha_short_read_and_nonzero_exit_abort(self):
        """Short or failed cat output must never complete or publish multipart data."""
        self.body = b'12'
        with patch.object(rpvolume, 'PART_SIZE', 3): asyncio.run(self.transfer())
        self.assertEqual(self.fake.mpu, {})
        self.assertNotIn('models/a.safetensors', self.fake.objs)

    def test_sync_now_recreates_only_explicitly(self):
        """A vanished billed volume must stay gone through automatic and plain sync rounds."""
        async def check():
            self.rest.vols = []
            self.ctl.state['missing'] = 2
            clock = [0]
            self.ctl.deps.now = lambda: clock[0]
            async def tick(seconds):
                clock[0] += seconds
                if clock[0] == 5: self.ctl.sync_now()
                if clock[0] == 10:
                    self.assertNotIn('POST', self.rest.events)
                    self.ctl.sync_now(recreate=True)
                if clock[0] >= 15: await self.ctl.aclose()
            with patch('rpvolume.asyncio.sleep', tick): await self.ctl.run_forever()
            self.assertIn('POST', self.rest.events)
        asyncio.run(check())

    def test_retry_success_after_transfer_error(self):
        """A transfer transport error must not poison the next good round's error state."""
        async def check():
            clock = [0]; original = self.fake.handler
            self.ctl.deps.now = lambda: clock[0]
            def lost_put(req):
                if req.method == 'PUT': raise httpx.ConnectError('offline')
                return original(req)
            self.fake.handler = lost_put
            async def tick(seconds):
                clock[0] += seconds
                if clock[0] == 25: self.fake.handler = original
                if clock[0] >= 40: await self.ctl.aclose()
            with patch('rpvolume.asyncio.sleep', tick): await self.ctl.run_forever()
            self.assertEqual(self.ctl._backoff, 0)
            self.assertEqual(self.ctl.sync_error, '')
            self.assertTrue(self.ctl.is_alias_ready('A'))
        asyncio.run(check())

    def test_aclose_interrupts_active_loop(self):
        """Shutdown must cancel an active stream so it cannot outlive its controller."""
        async def check():
            entered = asyncio.Event()
            async def status(bid, jid):
                entered.set(); await asyncio.Event().wait()
            self.ctl.state['fetch_job'] = {'id': 'old', 'bid': 'runpod:worker', 'ts': 0,
                'items': [{'path': 'models/a.safetensors', 'size': 5, 'url': 'https://example/a'}]}
            self.ctl.deps.fetch_status = status
            task = asyncio.create_task(self.ctl.run_forever())
            await entered.wait(); await self.ctl.aclose()
            await asyncio.sleep(0)
            done = task.done()
            if not done: task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(done)
            self.assertIsNotNone(self.ctl.state['fetch_job'])
        asyncio.run(check())

    def test_idle_tick_does_not_replan_ready_volume(self):
        """A ready volume must not pay for a fresh S3 listing every five seconds."""
        async def check():
            self.fake.objs['models/a.safetensors'] = b'12345'
            clock = [0]; self.ctl.deps.now = lambda: clock[0]
            async def tick(seconds):
                clock[0] += seconds
                if clock[0] >= 20: await self.ctl.aclose()
            with patch('rpvolume.asyncio.sleep', tick): await self.ctl.run_forever()
            listings = [p for method, p, auth in self.fake.calls if 'list-type' in p]
            self.assertEqual(len(listings), 2)
        asyncio.run(check())

    def test_delete_unknown_cannot_delete_needed_unowned_file(self):
        """A model required by an alias is not unknown merely because it lacks a manifest."""
        self.fake.objs['models/a.safetensors'] = b'12345'
        self.run_round()
        self.assertEqual(asyncio.run(self.ctl.delete_unknown(['models/a.safetensors'])), 0)
        self.assertTrue(self.ctl.is_alias_ready('A'))

    def test_url_without_share_gets_size_before_fetch(self):
        """A catalog-only model needs a bounded size even when the share cannot list it."""
        self.src = {}
        self.urls = {'models/a.safetensors': {'url': 'https://example/a'}}
        original = self.fake.handler
        def transport(request):
            if request.url.host == 'example':
                self.assertEqual(request.method, 'HEAD')
                self.assertNotIn('authorization', request.headers)
                return httpx.Response(200, headers={'content-length': '5'})
            if request.url.host == 'rest.runpod.io': return self.rest(request)
            return original(request)
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport))
        asyncio.run(self.transfer())
        self.assertEqual(self.payloads[0]['items'][0]['size'], 5)
        self.assertTrue(self.ctl.is_alias_ready('A'))

    def test_transient_url_failure_falls_back_on_third_attempt(self):
        """Transient fetch failures must consume exactly three attempts before LAN fallback."""
        self.result_ok = False
        self.urls = {'models/a.safetensors': {'url': 'https://example/a'}}
        asyncio.run(self.transfer())
        self.assertEqual(self.ctl.state['url_fallback'], {})
        asyncio.run(self.ctl._transfer_round())
        self.assertEqual(self.ctl.state['url_fallback'], {})
        asyncio.run(self.ctl._transfer_round())
        self.assertEqual(len(self.payloads), 3)
        self.assertEqual(self.ctl.state['url_fallback']['models/a.safetensors'], 'https://example/a')

    def test_resume_polls_then_clears_missing_job(self):
        """A saved queued job must be polled, and a 404 must release its reserved paths."""
        self.ctl.state['fetch_job'] = {'id': 'old', 'bid': 'runpod:worker', 'ts': 0,
            'items': [{'path': 'models/a.safetensors', 'size': 5, 'url': 'https://example/a'}]}
        async def check():
            calls = []
            async def status(bid, jid):
                calls.append(jid)
                return {'status': 'IN_QUEUE'} if len(calls) == 1 else None
            async def tick(seconds): self.assertEqual(seconds, 5)
            self.ctl.deps.fetch_status = status
            with patch('rpvolume.asyncio.sleep', tick): await self.ctl.resume()
            self.assertEqual(calls, ['old', 'old'])
            self.assertTrue(self.ctl._resumed)
            self.assertEqual(self.ctl._inflight, set())
            self.assertIsNone(self.saved[-1]['fetch_job'])
        asyncio.run(check())

    def test_no_backend_or_lan_reports_wait(self):
        """Missing endpoint or offline share must explain why sync cannot advance."""
        self.urls = {'models/a.safetensors': {'url': 'https://example/a'}}
        self.refs = []
        # Retain a plan from the referencing backend while it becomes unavailable.
        self.refs = [{'name': 'worker'}]
        async def check():
            await self.ctl.resume(); await self.ctl.plan_round()
            self.ctl._backends = []
            await self.ctl._transfer_round()
            self.assertIn('needs a RunPod endpoint on this volume', self.ctl.problems())
            self.ctl._backends = [{'name': 'worker'}]
            self.ctl.urls = {}
            self.ctl.deps.lan.usable = lambda: False
            await self.ctl._transfer_round()
            self.assertIn('waiting for LAN source (offline)', self.ctl.problems())
        asyncio.run(check())


class TransferReview(TransferRound):
    """Controller review of Task 8: each case once failed silently."""

    def test_loop_survives_a_failed_job(self):
        """run_fetch raises RuntimeError for a FAILED job (adapter _poll_rp). Uncaught, it
        ended run_forever for good — the volume never synced again, no error anywhere."""
        async def check():
            clock = [0]
            self.ctl.deps.now = lambda: clock[0]
            self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
            calls = []

            async def failing(bid, payload, on_id, *, max_wait=None):
                calls.append(clock[0])
                on_id('j1')                         # a FAILED job has an id
                raise RuntimeError('RunPod: job failed')
            self.ctl.deps.run_fetch = failing

            async def tick(seconds):
                clock[0] += seconds
                if clock[0] >= 200:
                    await self.ctl.aclose()
            with patch('rpvolume.asyncio.sleep', tick):
                await self.ctl.run_forever()
            self.assertTrue(calls)
            self.assertTrue(any('job failed' in line for line in self.logs))
            # later rounds ran: the saved job was settled through resume (fetch_status)
            self.assertIsNone(self.ctl.state['fetch_job'])
        asyncio.run(check())

    def test_lost_run_answer_holds_paths(self):
        """A /run whose answer was lost may have started a job with no id saved: its
        paths are held for the download budget plus queue allowance before another writer."""
        async def check():
            clock = [0]
            self.ctl.deps.now = lambda: clock[0]
            self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': 5}}
            ok = self.ctl.deps.run_fetch
            sent = []

            async def lost(bid, payload, on_id, *, max_wait=None):
                sent.append(clock[0])
                if len(sent) == 1:
                    raise httpx.ReadTimeout('answer lost')
                return await ok(bid, payload, on_id)
            self.ctl.deps.run_fetch = lost
            await self.ctl.resume()
            await self.ctl.plan_round()
            with self.assertRaises(httpx.ReadTimeout):
                await self.ctl._transfer_round()
            clock[0] = 60
            await self.ctl._transfer_round()
            self.assertEqual(len(sent), 1)
            clock[0] = 961
            await self.ctl._transfer_round()
            self.assertEqual(len(sent), 2)
        asyncio.run(check())

    def test_size_head_sends_hf_token_only_to_hf(self):
        """Without the token a gated HF file's size HEAD answers 401 and the file was
        given up for the LAN at once although the endpoint holds HF_TOKEN."""
        self.src = {}
        self.ctl.deps.hf_token = lambda: 'hf_tok'
        self.urls = {'models/a.safetensors': {'url': 'https://huggingface.co/r/resolve/main/a'},
                     'models/b.safetensors': {'url': 'https://example/b'}}
        self.needs = [need('A', [], catalog=list(self.urls))]
        seen = {}
        original = self.fake.handler

        def transport(request):
            if request.method == 'HEAD' and request.url.host in ('huggingface.co', 'example'):
                seen[request.url.host] = request.headers.get('authorization')
                return httpx.Response(200, headers={'content-length': '5'})
            if request.url.host == 'rest.runpod.io':
                return self.rest(request)
            return original(request)
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport))
        asyncio.run(self.transfer())
        self.assertEqual(seen, {'huggingface.co': 'Bearer hf_tok', 'example': None})

    def test_file_above_job_bytes_goes_alone(self):
        """JOB_BYTES bounds a batch, not a file: a 30 GB model waited forever."""
        big = rpvolume.JOB_BYTES + 1
        self.urls = {'models/a.safetensors': {'url': 'https://example/a', 'size': big}}
        self.src['models/a.safetensors'] = big
        self.ctl.cfg['max_size_gb'] = 100
        self.result_ok = False
        asyncio.run(self.transfer())
        self.assertEqual([i['size'] for i in self.payloads[0]['items']], [big])


class FixRound(unittest.TestCase):
    """Final review regressions pin recovery, bounded work and accurate billing."""
    setUp = TransferRound.setUp
    manifest = PlanRound.manifest
    run_round = PlanRound.run_round

    async def transfer(self):
        await self.ctl.resume()
        await self.ctl.plan_round()
        await self.ctl._transfer_round()

    def test_sync_now_releases_failures_except_held_paths(self):
        """An operator retry must release stale failures without racing a saved writer."""
        p, held = 'models/a.safetensors', 'models/held'
        self.ctl.state['blocked'] = {p: 'share file changing?', held: 'final: HTTP 401',
                                    'models/cap': 'needs 50 GB, limit 40 GB'}
        self.ctl.state['fetch_job'] = {'items': [{'path': held}]}
        self.ctl.attempts = {p: 3, held: 3}
        self.ctl._stream_failures = {p: 2, held: 2}
        self.ctl.state['url_fallback'] = {p: 'https://e/a', held: 'https://e/h'}
        self.ctl.state['url_fallback_why'] = {p: 'failed', held: 'failed'}
        forgotten = []
        self.ctl.deps.lan.forget_sha = lambda path, size: forgotten.append(path)
        self.ctl.sync_now()
        self.assertNotIn(p, self.ctl.state['blocked'])
        self.assertEqual(set(self.ctl.state['blocked']), {held, 'models/cap'})
        self.assertEqual(self.ctl.attempts, {held: 3})
        self.assertEqual(self.ctl._stream_failures, {held: 2})
        self.assertEqual(self.ctl.state['url_fallback'], {held: 'https://e/h'})
        self.assertEqual(self.ctl.state['url_fallback_why'], {held: 'failed'})
        self.assertEqual(forgotten, [p])
        self.assertEqual(len(self.saved), 1)

    def test_present_file_drops_transfer_block(self):
        """A manually repaired file must open its alias even with a persisted failure."""
        p = 'models/a.safetensors'
        self.ctl.state['blocked'][p] = 'final: HTTP 401'
        self.fake.objs[p] = self.body
        self.run_round()
        self.assertTrue(self.ctl.is_alias_ready('A'))
        self.assertNotIn(p, self.ctl.state['blocked'])

    def test_sha_mismatch_forgets_cached_share_hash(self):
        """A stale LAN hash must not turn a stable share file into a permanent block."""
        forgotten = []
        self.ctl.deps.lan.forget_sha = lambda path, size: forgotten.append(path)
        async def stale(path, size): return '0' * 64
        self.ctl.deps.lan.sha256 = stale
        asyncio.run(self.transfer())
        self.assertEqual(forgotten, ['models/a.safetensors'])

    def test_waiting_ticks_do_no_io_or_saves(self):
        """An offline share must not cause S3 lists, store writes or console polling every tick."""
        self.ctl.deps.lan.usable = lambda: False
        asyncio.run(self.transfer())
        calls, saves = len(self.fake.calls), len(self.saved)
        for _ in range(10): asyncio.run(self.ctl._transfer_round())
        self.assertEqual(len(self.fake.calls), calls)
        self.assertEqual(len(self.saved), saves)
        self.assertEqual(self.ctl.view()['phase'], 'waiting')
        self.run_round()
        self.assertEqual(len(self.saved), saves)

    def test_job_timeout_does_not_consume_item_attempts(self):
        """Queue or job failures must back off without sacrificing untouched URL files."""
        self.src = {}
        self.urls = {'models/a.safetensors': {'url': 'https://e/a', 'size': 5}}
        self.manifest({'models/a.safetensors': {'size': 5}})
        async def failed(bid, payload, on_id, *, max_wait=None):
            on_id('j'); return dict(status='TIMED_OUT', output=None)
        self.ctl.deps.run_fetch = failed
        for _ in range(3):
            with self.assertRaises(RuntimeError): asyncio.run(self.transfer())
        self.assertEqual(self.ctl.attempts, {})
        self.assertEqual(self.ctl.state['url_fallback'], {})
        self.assertIsNone(self.ctl.state['fetch_job'])
        self.assertIn('fetch jobs paused after 3 failed jobs (last: TIMED_OUT) — Sync now retries', self.ctl.problems())

    def test_head_405_unknown_size_cached_and_fetched_alone(self):
        """GET-only URLs must reach the worker and failed HEADs must not repeat every tick."""
        self.src = {}
        p = 'models/a.safetensors'
        self.urls = {p: {'url': 'https://example/a'}}
        heads = []
        def transport(req):
            if req.url.host == 'example':
                heads.append(req); return httpx.Response(405)
            return self.rest(req) if req.url.host == 'rest.runpod.io' else self.fake.handler(req)
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.run_round()
        entry = self.ctl.plan['fetch'][0]
        clock = [0]; self.ctl.deps.now = lambda: clock[0]
        for _ in range(2):
            out = asyncio.run(self.ctl._fetch_entry(entry))
            self.assertIsNotNone(out)
            self.assertIsNone(out['size'])
        self.assertEqual(len(heads), 1)
        clock[0] = 601
        asyncio.run(self.ctl._fetch_entry(entry))
        self.assertEqual(len(heads), 2)
        self.ctl.sync_now()
        asyncio.run(self.transfer())
        self.assertEqual(len(heads), 3)
        self.assertIsNone(self.payloads[0]['items'][0]['size'])
        self.assertEqual(next(s['fetch_job']['budget_s'] for s in self.saved if s['fetch_job']), 14400)
        self.assertEqual(self.ctl.manifest[p]['size'], 5)
        self.assertTrue(self.ctl.is_alias_ready('A'))

    def test_fetch_budget_and_aliases_saved_before_poll(self):
        """Fetch execution limits must reflect download bytes and retain alias ownership for resume."""
        seen = []
        async def pending(bid, payload, on_id, *, max_wait=None):
            on_id('j'); seen.append((max_wait, copy.deepcopy(self.saved[-1]['fetch_job'])))
            return {'status': 'IN_PROGRESS'}
        self.ctl.deps.run_fetch = pending
        entries = [dict(path='models/a', url='https://e/a', size=20_000_000_000, aliases=['A'])]
        asyncio.run(self.ctl._fetch_batch('runpod:worker', entries))
        self.assertEqual(seen[0][0], 1300)
        self.assertEqual(seen[0][1]['budget_s'], 1300)
        self.assertEqual(seen[0][1]['items'][0]['aliases'], ['A'])

    def test_one_unit_refreshes_ready_without_listing(self):
        """Alias A must become routable before alias B's multi-hour LAN upload starts."""
        self.src = {'models/a.safetensors': 5, 'models/b': 5}
        self.needs = [need('A', [], catalog=['models/a.safetensors']), need('B', [], catalog=['models/b'])]
        async def check():
            await self.ctl.resume(); await self.ctl.plan_round()
            listings = len([c for c in self.fake.calls if 'list-type' in c[1]])
            await self.ctl._transfer_round()
            self.assertTrue(self.ctl.is_alias_ready('A'))
            self.assertNotIn('models/b', self.fake.objs)
            self.assertEqual(len([c for c in self.fake.calls if 'list-type' in c[1]]), listings)
            await self.ctl._transfer_round()
            self.assertTrue(self.ctl.is_alias_ready('B'))
        asyncio.run(check())

    def test_remaining_need_does_not_bill_present_or_blocked_files(self):
        """A late URL HEAD must not grow for a LAN file already uploaded or a blocked path."""
        x, y, z = 'models/x', 'models/y', 'models/z'
        def transport(req):
            if req.url.host == 'example': return httpx.Response(200, headers={'content-length': '1000000000'})
            return self.rest(req) if req.url.host == 'rest.runpod.io' else self.fake.handler(req)
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.ctl.plan = {'fetch': [dict(path=x, size=5_000_000_000, source='lan'),
            dict(path=y, size=None, source='url', url='https://example/y'),
            dict(path=z, size=30_000_000_000, source='lan')], 'prune': [], 'per_alias': {}}
        self.ctl.dest = {x: 5_000_000_000}
        self.ctl.state['blocked'][z] = 'share file changing?'
        self.assertIsNotNone(asyncio.run(self.ctl._fetch_entry(self.ctl.plan['fetch'][1])))
        self.assertNotIn('PATCH', self.rest.events)
        self.assertEqual(self.ctl.state['size_gb'], 10)

    def test_stream_hash_runs_in_thread(self):
        """Large LAN chunks must not stall request handling while hashing."""
        self.body = b'x' * (1024 * 1024)
        self.src['models/a.safetensors'] = len(self.body)
        # Avoid putting a megabyte in subprocess argv.
        self.ctl.deps.lan.cat_argv = lambda p, off: [sys.executable, '-c', "import sys; sys.stdout.buffer.write(b'x' * 1048576)"]
        orig = asyncio.to_thread
        calls = []
        async def spy(fn, *args, **kwargs):
            calls.append(fn); return await orig(fn, *args, **kwargs)
        with patch('rpvolume.asyncio.to_thread', spy): asyncio.run(self.transfer())
        self.assertTrue(any(getattr(fn, '__name__', '') == 'update' for fn in calls))

    def test_part_bytes_visible_and_cleanup_respects_final_holds(self):
        """Part leftovers occupy capacity and cleanup must not delete an active download's part."""
        p = 'models/a.safetensors'
        self.fake.objs[p + '.gw-part'] = b'123'
        self.fake.objs['models/old.gw-part'] = b'12'
        self.run_round()
        self.assertEqual(self.ctl.view()['used_bytes'], 5)
        self.assertEqual(self.ctl.view()['leftovers'], [[p + '.gw-part', 3], ['models/old.gw-part', 2]])
        for kind in ('inflight', 'fetch_job', 'ghost', 'mpu'):
            with self.subTest(hold=kind):
                self.fake.objs['models/old.gw-part'] = b'12'
                self.ctl._inflight.clear(); self.ctl._ghost.clear()
                self.ctl.state.update(fetch_job=None, mpu=[])
                if kind == 'inflight': self.ctl._inflight.add(p)
                elif kind == 'fetch_job': self.ctl.state['fetch_job'] = {'items': [{'path': p}]}
                elif kind == 'ghost': self.ctl._ghost[p] = self.ctl.deps.now() + 100
                else: self.ctl.state['mpu'] = [{'key': p, 'upload_id': 'u'}]
                count = asyncio.run(self.ctl.delete_unknown([p + '.gw-part', 'models/old.gw-part']))
                self.assertEqual(count, 1)
                self.assertIn(p + '.gw-part', self.fake.objs)

    def test_ghost_only_uncertain_submit_and_survives_restart(self):
        """Definite pre-submit failures must not hold paths, while uncertain jobs survive restart for their TTL."""
        self.refs[0]['queue_max_s'] = 900
        entry = dict(path='models/a', size=5, url='https://e/a', aliases=['A'])
        async def check():
            async def definite(*args, **kwargs): raise httpx.ConnectError('not connected')
            self.ctl.deps.run_fetch = definite
            with self.assertRaises(httpx.ConnectError): await self.ctl._fetch_batch('runpod:worker', [entry])
            self.assertEqual(self.ctl._ghost, {})
            async def uncertain(*args, **kwargs): raise httpx.ReadTimeout('lost')
            self.ctl.deps.run_fetch = uncertain
            self.ctl.deps.now = lambda: 100
            with self.assertRaises(httpx.ReadTimeout): await self.ctl._fetch_batch('runpod:worker', [entry])
            self.assertEqual(self.saved[-1]['ghost'], {'models/a': 1660})
            self.ctl.deps.load_state = lambda name: self.saved[-1]
            restarted = rpvolume.VolumeController('models', self.ctl.cfg, self.ctl.deps)
            self.assertEqual(restarted._ghost, {'models/a': 1660})
        asyncio.run(check())

    def test_resume_uses_saved_budget_and_preserves_aliases(self):
        """A legitimately running long fetch must not be cancelled at the old 30-minute deadline."""
        p = 'models/a.safetensors'
        job = dict(id='old', bid='runpod:worker', ts=0, budget_s=3600,
                   items=[dict(path=p, size=5, url='https://e/a', aliases=['A'])])
        self.ctl.state['fetch_job'] = job
        async def check():
            clock, cancels = [0], []
            self.ctl.deps.now = lambda: clock[0]
            async def status(bid, jid):
                if clock[0] < 3600: return {'status': 'IN_PROGRESS'}
                self.fake.objs[p] = b'12345'
                return dict(status='COMPLETED', output={'results': [dict(path=p, ok=True, sha256='abc')]})
            async def cancel(bid, jid): cancels.append(jid); return True
            async def tick(seconds): clock[0] += 1800
            self.ctl.deps.fetch_status, self.ctl.deps.fetch_cancel = status, cancel
            with patch('rpvolume.asyncio.sleep', tick): await self.ctl.resume()
            self.assertEqual(cancels, [])
            self.assertEqual(self.ctl.manifest[p]['aliases'], ['A'])
        asyncio.run(check())

    def test_expired_job_reports_lost_result(self):
        """Expired RunPod results must release writers with a visible cost/result-loss warning."""
        self.ctl.state['fetch_job'] = dict(id='old', bid='runpod:worker', ts=0, items=[])
        async def expired(bid, jid): return None
        self.ctl.deps.fetch_status = expired
        asyncio.run(self.ctl.resume())
        self.assertIn('fetch job old result lost (expired at RunPod) — files are re-checked', self.ctl.problems())
        self.assertTrue(any('result lost' in line for line in self.logs))

    def test_new_head_size_obeys_capacity_ceiling(self):
        """A formerly sizeless file must not bypass the growth ceiling after HEAD succeeds."""
        self.src = {}
        p = 'models/a.safetensors'
        self.urls = {p: {'url': 'https://example/a'}}
        self.ctl.cfg['max_size_gb'] = 12
        async def submitted(bid, payload, on_id, *, max_wait=None):
            self.payloads.append(payload); on_id('j')
            return {'status': 'IN_PROGRESS'}
        self.ctl.deps.run_fetch = submitted
        def transport(req):
            if req.url.host == 'example': return httpx.Response(200, headers={'content-length': '20000000000'})
            return self.rest(req) if req.url.host == 'rest.runpod.io' else self.fake.handler(req)
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport))
        asyncio.run(self.transfer())
        self.assertEqual(self.payloads, [])
        self.assertIn('limit 12 GB', self.ctl.state['blocked'][p])
        self.assertEqual(self.rest.vols[0]['size'], 12)

    def test_sync_now_forgets_persisted_lan_failure(self):
        """After restart a persisted LAN failure must also evict the share's cached hash."""
        p = 'models/a.safetensors'
        self.ctl.state['blocked'][p] = 'share file changing?'
        import hostctl
        cache = {'host': 'src@10.0.0.2', 'files': {p: [5, 'a' * 64]}}
        persisted = []
        self.ctl.deps.lan = hostctl.LanSource('.', lambda: 'src@10.0.0.2',
            load_sha=lambda: cache, save_sha=persisted.append)
        self.assertEqual(self.ctl.deps.lan.known_sha(p, 5), 'a' * 64)
        self.ctl.sync_now()
        self.assertIsNone(self.ctl.deps.lan.known_sha(p, 5))
        self.assertEqual(persisted[-1]['files'], {})

    def test_lost_result_problem_survives_waiting_tick(self):
        """The loop must not erase the lost-result warning before the card can render it."""
        self.ctl.state['fetch_job'] = dict(id='old', bid='runpod:worker', ts=0, items=[])
        self.ctl.deps.lan.usable = lambda: False
        async def expired(bid, jid): return None
        self.ctl.deps.fetch_status = expired
        asyncio.run(self.transfer())
        self.assertTrue(any('result lost' in p for p in self.ctl.problems()))

    def test_job_output_mount_error_is_not_an_item_failure(self):
        """An unattached endpoint must explain its job failure without consuming URL retries."""
        self.urls = {'models/a.safetensors': {'url': 'https://e/a'}}
        async def unmounted(bid, payload, on_id, *, max_wait=None):
            on_id('j')
            return dict(status='COMPLETED', output={'error': 'network volume not mounted at /runpod-volume'})
        self.ctl.deps.run_fetch = unmounted
        with self.assertRaisesRegex(RuntimeError, 'network volume not mounted'): asyncio.run(self.transfer())
        self.assertEqual(self.ctl.attempts, {})
        self.assertEqual(self.ctl.state['url_fallback'], {})
        self.assertIsNone(self.ctl.state['fetch_job'])

    def test_blocked_file_does_not_grow_on_full_plan(self):
        """Persisted transfer failures must not buy capacity for files that cannot be written."""
        p = 'models/a.safetensors'
        self.src[p] = 30_000_000_000
        self.ctl.state['blocked'][p] = 'share file changing?'
        self.run_round()
        self.assertNotIn('PATCH', self.rest.events)
        self.assertEqual(self.ctl.view()['phase'], 'waiting')

    def test_resume_alias_ownership_survives_temporary_alias_block(self):
        """A resumed download must retain the alias hold that protects it from pruning."""
        p = 'models/a.safetensors'
        self.ctl.state['fetch_job'] = dict(id='old', bid='runpod:worker', ts=0,
            items=[dict(path=p, size=5, url='https://e/a', aliases=['A'])])
        async def finished(bid, jid):
            self.fake.objs[p] = b'12345'
            return dict(status='COMPLETED', output={'results': [dict(path=p, ok=True, sha256='abc')]})
        self.ctl.deps.fetch_status = finished
        asyncio.run(self.ctl.resume())
        self.assertEqual(self.ctl.manifest[p]['aliases'], ['A'])
        self.src = {}
        self.needs = [need('A', [], catalog=['models/unknown'])]
        self.run_round()
        self.assertNotIn(p, self.ctl.plan['prune'])
        self.assertTrue(self.ctl.plan['held'])

    def test_failed_job_salvages_published_catalog_file(self):
        """A timed-out job must not repeatedly download its published files (repro4)."""
        p = 'models/a.safetensors'
        self.src = {}
        self.urls = {p: {'url': 'https://example/a'}}
        def transport(req):
            if req.url.host == 'example': return httpx.Response(405)
            return self.rest(req) if req.url.host == 'rest.runpod.io' else self.fake.handler(req)
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport))
        async def failed(bid, payload, on_id, *, max_wait=None):
            self.payloads.append(payload)
            on_id('j')
            self.fake.objs[p] = b'12345'
            return dict(status='TIMED_OUT', output=None, executionTime=3600000)
        self.ctl.deps.run_fetch = failed
        with self.assertRaises(RuntimeError): asyncio.run(self.transfer())
        self.assertEqual(self.ctl.manifest[p]['sha256'], '')
        self.assertEqual(self.ctl.manifest[p]['source'], 'url')
        self.assertTrue(self.ctl.is_alias_ready('A'))
        self.assertEqual(self.ctl.state['job_failures'], 1)
        self.assertEqual(self.ctl.state['sync_cost_usd'], 2)

    def test_job_failures_pause_urls_but_allow_lan_and_sync_now(self):
        """Persistent endpoint failures must stop billing after three jobs, across restart."""
        p = 'models/a.safetensors'
        self.urls = {p: {'url': 'https://example/a', 'size': 5}}
        async def failed(bid, payload, on_id, *, max_wait=None):
            self.payloads.append(payload)
            on_id('j')
            return dict(status='TIMED_OUT', output=None)
        self.ctl.deps.run_fetch = failed
        asyncio.run(self.ctl.resume())
        asyncio.run(self.ctl.plan_round())
        for _ in range(8):
            try: asyncio.run(self.ctl._transfer_round())
            except RuntimeError: pass
        self.assertEqual(len(self.payloads), 3)
        self.assertEqual(self.saved[-1]['job_failures'], 3)
        self.assertIn('fetch jobs paused after 3 failed jobs (last: TIMED_OUT) — Sync now retries', self.ctl.problems())
        self.src['models/b'] = 5
        self.needs = [need('A', [], catalog=list(self.src))]
        asyncio.run(self.ctl.plan_round())
        asyncio.run(self.ctl._transfer_round())
        self.assertEqual(self.fake.objs['models/b'], b'12345')
        self.ctl.sync_now()
        self.assertEqual(self.ctl.state['job_failures'], 0)

    def test_partial_bytes_do_not_order_permanent_growth(self):
        """An 8 GB file with a 4 GB prefix fits a 10 GB volume (repro4)."""
        p = 'models/x.safetensors'
        self.src = {p: 8 * 10**9}
        self.needs = [need('A', [], catalog=[p])]
        self.urls = {p: {'url': 'https://example/x'}}
        self.fake.objs[p + '.gw-part'] = SizedBody(4 * 10**9)
        self.run_round()
        self.assertEqual(self.ctl.state['size_gb'], 10)
        self.assertNotIn('PATCH', self.rest.events)
        self.assertNotIn(p, self.ctl.state['blocked'])

    def test_paused_jobs_survive_restart_and_item_results_reset(self):
        """A gateway restart must not release a persisted billing pause."""
        self.ctl.state.update(job_failures=3, job_failure_reason='TIMED_OUT')
        self.ctl._save()
        self.ctl.deps.load_state = lambda name: self.saved[-1]
        restarted = rpvolume.VolumeController(self.ctl.name, self.ctl.cfg, self.ctl.deps)
        self.assertEqual(restarted.state['job_failures'], 3)
        self.assertIn('Sync now retries', ' '.join(restarted.problems()))
        job = dict(bid='runpod:worker', items=[dict(path='models/a', size=5)])
        async def finish():
            async with self.ctl.deps.client_factory() as client:
                await self.ctl._finish_fetch(self.ctl._client_s3(client), job,
                    dict(output={'results': [dict(path='models/a', ok=False, error='temporary')]}))
        asyncio.run(finish())
        self.assertEqual(self.ctl.state['job_failures'], 0)
        self.assertEqual(self.saved[-1]['job_failures'], 0)

    def test_link_item_results_reset_job_failure_counter(self):
        """A successful link job breaks the sequence of endpoint-wide failures too."""
        self.ctl.state.update(job_failures=3, job_failure_reason='TIMED_OUT')
        async def finish():
            async with self.ctl.deps.client_factory() as client:
                await self.ctl._finish_links(self.ctl._client_s3(client),
                    dict(bid='runpod:worker', items=[]),
                    dict(status='COMPLETED', output={'results': [dict(path='hf-cache/a', ok=True)]}))
        asyncio.run(finish())
        self.assertEqual(self.ctl.state['job_failures'], 0)
        self.assertEqual(self.saved[-1]['job_failures'], 0)

    def test_head_learned_size_discounts_partial_growth(self):
        """Learning a catalog-only size must use the same partial-aware capacity check."""
        p = 'models/x.safetensors'
        self.src = {}
        self.needs = [need('A', [], catalog=[p])]
        self.urls = {p: {'url': 'https://example/x'}}
        self.fake.objs[p + '.gw-part'] = SizedBody(4 * 10**9)
        def transport(req):
            if req.url.host == 'example': return httpx.Response(200, headers={'content-length': str(8 * 10**9)})
            return self.rest(req) if req.url.host == 'rest.runpod.io' else self.fake.handler(req)
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.run_round()
        asyncio.run(self.ctl._fetch_entry(self.ctl.plan['fetch'][0]))
        self.assertEqual(self.ctl.state['size_gb'], 10)
        self.assertNotIn('PATCH', self.rest.events)
        self.assertNotIn(p, self.ctl.state['blocked'])

    def test_poll_exception_preserves_real_mount_problem(self):
        """An adapter exception after submit must show the worker's actual mount problem."""
        mount = 'network volume not mounted at /runpod-volume'
        self.urls = {'models/a.safetensors': {'url': 'https://e/a', 'size': 5}}
        async def failed(bid, payload, on_id, *, max_wait=None):
            on_id('j')
            raise RuntimeError('RunPod: ' + mount)
        self.ctl.deps.run_fetch = failed
        with self.assertRaises(RuntimeError): asyncio.run(self.transfer())
        self.assertIn(mount, ' '.join(self.ctl.problems()))
        self.assertEqual(self.ctl.state['fetch_job']['id'], 'j')

    def test_real_mount_error_and_exception_are_scrubbed(self):
        """RunPod top-level errors and poll exceptions must explain failed jobs safely."""
        mount = 'network volume not mounted at /runpod-volume'
        for status in ({'status': 'FAILED', 'error': mount},
                       {'error': RuntimeError('RunPod: ' + mount)}):
            with self.assertRaisesRegex(RuntimeError, mount): self.ctl._job_failed(status)
        with self.assertRaises(RuntimeError):
            self.ctl._job_failed({'error': 'x' * 220 + ' https://e/a?secret=z'})
        self.assertLessEqual(len(self.ctl._fetch_problem.removeprefix('fetch job failed: ')), 200)
        self.assertNotIn('https://', self.ctl._fetch_problem)

    def test_job_level_error_does_not_expose_signed_url(self):
        """Job-wide error text must not expose RunPod's stored signed download URL in the card or log."""
        url = 'https://e/a?token=private-download-token'
        self.urls = {'models/a.safetensors': {'url': url}}
        async def failed(bid, payload, on_id, *, max_wait=None):
            on_id('j')
            return dict(status='FAILED', output={'error': 'download failed at ' + url})
        self.ctl.deps.run_fetch = failed
        with self.assertRaises(RuntimeError): asyncio.run(self.transfer())
        self.assertNotIn('private-download-token', json.dumps(self.ctl.view()))

    def _two_sizeless_urls(self):
        self.src = {}
        self.urls = {p: {'url': 'https://example/' + p} for p in ('models/a', 'models/b')}
        self.needs = [need('A', [], catalog=list(self.urls))]
        def transport(req):
            if req.url.host == 'example': return httpx.Response(200, headers={'content-length': '6000000000'})
            return self.rest(req) if req.url.host == 'rest.runpod.io' else self.fake.handler(req)
        self.ctl.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(transport))
        async def submitted(bid, payload, on_id, *, max_wait=None):
            self.payloads.append(payload); on_id('j')
            return {'status': 'IN_PROGRESS'}
        self.ctl.deps.run_fetch = submitted

    def test_multiple_head_sizes_are_counted_together(self):
        """Individually fitting sizeless URLs must not overfill a volume when batched together."""
        self._two_sizeless_urls()
        asyncio.run(self.transfer())
        self.assertEqual(self.ctl.state['size_gb'], 14)
        self.assertEqual([i['size'] for i in self.payloads[0]['items']], [6_000_000_000, 6_000_000_000])

    def test_later_head_capacity_block_removes_earlier_batch_item(self):
        """A capacity decision made while preparing a batch must protect every newly blocked path."""
        self.ctl.cfg['max_size_gb'] = 10
        self._two_sizeless_urls()
        asyncio.run(self.transfer())
        self.assertEqual(self.payloads, [])
        self.assertEqual(set(self.ctl.state['blocked']), {'models/a', 'models/b'})
