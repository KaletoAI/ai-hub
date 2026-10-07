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
