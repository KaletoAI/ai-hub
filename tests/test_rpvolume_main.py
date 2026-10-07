"""Volume wiring must keep routing closed and persisted writers recoverable."""
import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.test_runpod_adapter import _main
import store


class VolumeMain(unittest.TestCase):
    def setUp(self):
        self.m = _main()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved_store = (store._DB_PATH, store._active, store._MASTER_KEY)
        def restore_store():
            store._DB_PATH, store._active, store._MASTER_KEY = saved_store
        self.addCleanup(restore_store)
        store.init(self.tmp.name + '/store.db')
        for attr, value in [('runpod_volumes', {}), ('volume_controllers', {}),
                            ('_vol_tasks', {}), ('_vol_warned', set())]:
            p = patch.object(self.m, attr, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        self.cfg = dict(datacenter='EU-RO-1', size_gb=50, max_size_gb=200)

    def test_gate_blocks_until_ready(self):
        """An attached endpoint cannot serve models before its volume plan is ready."""
        b = dict(type='runpod', name='rp1', volume='models')
        self.assertEqual(self.m.modelsync_gate(b, 'a'), 'RunPod volume models is not set up yet')
        c = Mock()
        c.is_alias_ready.return_value = False
        c.alias_status.return_value = 'syncing'
        self.m.volume_controllers['models'] = c
        self.assertEqual(self.m.modelsync_gate(b, 'a'), 'syncing')
        c.is_alias_ready.return_value = True
        self.assertIsNone(self.m.modelsync_gate(b, 'a'))

    def test_gate_ignores_runpod_without_volume(self):
        """M1 endpoints without a managed volume remain routable."""
        self.assertIsNone(self.m.modelsync_gate(dict(type='runpod'), 'a'))

    def test_save_volume_refusals(self):
        """Invalid edits cannot change DC or lower the ceiling below the billed size."""
        m = self.m
        self.assertTrue(m.save_volume('BAD', self.cfg, True))
        self.assertEqual(m.save_volume('models', self.cfg, True), '')
        self.assertTrue(m.save_volume('models', self.cfg, True))
        c = m.volume_controllers['models']
        c.state.update(id='v1', size_gb=70)
        self.assertTrue(m.save_volume('models', dict(self.cfg, datacenter='US-CA-2'), False))
        # created and grown to 70: the start size (50) is no shrink — an edit of the
        # ceiling must still save (the browser check found every edit refused)
        self.assertEqual(m.save_volume('models', self.cfg, False), '')
        self.assertIn('only grows', m.save_volume('models', dict(self.cfg, max_size_gb=60), False))
        self.assertTrue(m.save_volume('models', dict(self.cfg, max_size_gb=20), False))
        entry = dict(self.cfg, size_gb=70, max_size_gb=300)
        self.assertEqual(m.save_volume('models', entry, False), '')
        # the start size of a created volume is kept; the new ceiling applies at once
        self.assertEqual(c.cfg, dict(self.cfg, max_size_gb=300))

    def test_runpod_s3_secret_format(self):
        """Malformed S3 credentials are refused without leaking their value."""
        for token in ['x', ':y', 'x:', 'x:y:z', 'x: y', 'x:y\n']:
            self.assertEqual(self.m.save_provider_token('runpod_s3', token),
                             'expected <access key id>:<secret>')
        self.assertEqual(self.m.save_provider_token('runpod_s3', 'x:y'), '')
        self.assertTrue(self.m.provider_tokens_info()['runpod_s3'])
        self.assertEqual(self.m._volume_deps('models').creds(), ('', 'x', 'y'))

    def test_backend_volume_must_exist(self):
        """A dangling volume selection must be refused by the admin save seam."""
        self.assertEqual(self.m.volume_field_refusal(''), '')
        self.assertTrue(self.m.volume_field_refusal('missing'))
        self.m.save_volume('models', self.cfg, True)
        self.assertEqual(self.m.volume_field_refusal('models'), '')

    def test_alias_needs_for_runpod_bid(self):
        """RunPod ComfyUI candidates must participate in volume model planning."""
        cand = dict(backend='rp1', workflow_json={})
        with patch.object(self.m, '_comfy_alias_cands', return_value=[('a', cand)]):
            self.assertEqual([n.alias for n in self.m.service_alias_needs('runpod:rp1', [])], ['a'])

    def test_bulk_state_and_last_good_read(self):
        """Settings overlays cannot expose bulk state or erase controllers on a read error."""
        self.m.save_volume('models', self.cfg, True)
        self.m._volume_save_state('models', {'id': 'v1'})
        self.m._volume_save_state('other', {'id': 'v2'})
        self.assertEqual(self.m._volume_load_state('models'), {'id': 'v1'})
        self.assertNotIn('runpod_volumes', store.get_settings())
        self.assertNotIn('runpod_volume_state', store.get_settings())
        with patch.object(store, 'get_setting', side_effect=RuntimeError()):
            self.assertIn('models', self.m._load_runpod_volumes())

    def test_live_adapter_and_backend_filter(self):
        """Fetch recovery must use the live adapter and only enabled volume endpoints."""
        ad = Mock(spec=self.m.adapters.RunpodAdapter, run_op=AsyncMock(return_value={'status': 'COMPLETED'}),
                  job_status=AsyncMock(return_value=None), cancel_runpod_id=AsyncMock(return_value=True))
        b = dict(name='rp1', type='runpod', volume='models', max_wait=123)
        with patch.object(self.m, 'backends', [b, dict(b, name='off', enabled=False)]), \
             patch.object(self.m, 'backend_adapters', {'runpod:rp1': ad}):
            d = self.m._volume_deps('models')
            self.assertEqual(d.backends('models'), [b])
            on_id = Mock()
            asyncio.run(d.run_fetch('runpod:rp1', {'op': 'fetch'}, on_id))
            ad.run_op.assert_awaited_once_with({'op': 'fetch'}, 123.0, on_id)
            asyncio.run(d.fetch_status('runpod:rp1', 'j'))
            asyncio.run(d.fetch_cancel('runpod:rp1', 'j'))
            with self.assertRaises(RuntimeError):
                asyncio.run(d.fetch_status('runpod:missing', 'j'))

    def test_resume_error_keeps_background_recovery(self):
        """A transient boot recovery error cannot leave a volume undriven forever."""
        async def run():
            c = Mock(resume=AsyncMock(side_effect=RuntimeError('temporary')),
                     run_forever=AsyncMock())
            self.m._vol_run('models', c)
            await asyncio.gather(*self.m._vol_tasks['models'], return_exceptions=True)
            c.run_forever.assert_awaited_once()
        asyncio.run(run())

    def test_retirement_keeps_unsettled_writers(self):
        """Removing a store entry cannot discard recovery records for active writers."""
        self.m.save_volume('models', self.cfg, True)
        c = self.m.volume_controllers['models']
        store.set_settings({'runpod_volumes': {}})
        for state in [{'fetch_job': {'id': 'j'}, 'mpu': []},
                      {'fetch_job': None, 'mpu': [{'key': 'models/x'}]}]:
            c.state.update(state)
            self.m.sync_volume_controllers()
            self.assertIs(self.m.volume_controllers['models'], c)
        c.state.update(fetch_job=None, mpu=[])
        self.m.sync_volume_controllers()
        self.assertNotIn('models', self.m.volume_controllers)

    def test_boot_and_shutdown(self):
        """Boot recovers before running, and shutdown closes every volume controller."""
        async def run():
            c = Mock(resume=AsyncMock(), run_forever=AsyncMock(), aclose=AsyncMock())
            self.m.volume_controllers['models'] = c
            with patch.object(self.m, 'host_controllers', {}), \
                 patch.object(self.m, '_host_tasks', {}), \
                 patch.object(self.m, '_modelsrc_prepare'), \
                 patch.object(self.m, '_modelsrc_key_task', None), \
                 patch.object(self.m, '_hosts_booted', False):
                self.m._hosts_boot()
                await asyncio.gather(*self.m._vol_tasks['models'])
                c.resume.assert_awaited_once()
                c.run_forever.assert_awaited_once()
                await self.m._hosts_shutdown()
                c.aclose.assert_awaited_once()
        asyncio.run(run())

    def test_rebuild_preserves_volume_and_starts_late_controller(self):
        """Backend rebuilds must keep volume attachments and start post-boot volumes."""
        b = dict(name='rp1', type='runpod', volume='models', url='https://api.runpod.ai/v2/e')
        store.upsert_backend(b)
        with patch.object(self.m, 'config_backends', []), \
             patch.object(self.m, 'sync_host_controllers'), \
             patch.object(self.m, 'apply_hosts'), \
             patch.object(self.m, 'backends', []), \
             patch.object(self.m, 'backend_hosts', {}), \
             patch.object(self.m, 'host_backends', {}), \
             patch.object(self.m, 'rebuild_route_index'), \
             patch.object(self.m, '_hosts_booted', True), \
             patch.object(self.m, '_vol_run') as start:
            store.set_settings({'runpod_volumes': {'models': self.cfg}})
            self.m.rebuild_backends()
            self.assertEqual(self.m.backends[0]['volume'], 'models')
            start.assert_called_once_with('models', self.m.volume_controllers['models'])

    def test_size_boundaries_and_delete_references(self):
        """Volume validation cannot bypass provider limits or delete disabled attachments."""
        for size, maximum in [(9, 50), (4001, 4001), (50, 4001), (True, 50), ('50', 200)]:
            self.assertTrue(self.m.save_volume('models', dict(self.cfg, size_gb=size,
                                                            max_size_gb=maximum), True))
        self.m.save_volume('models', self.cfg, True)
        with patch.object(self.m, 'backends', [dict(type='runpod', volume='models', enabled=False)]):
            self.assertTrue(asyncio.run(self.m.delete_volume('models')))

    def test_delete_never_created_volume(self):
        """A volume never created at RunPod (no id saved) could not be deleted: the
        fresh-list lookup never finds it, so its entry stayed in the console for good."""
        self.assertEqual(self.m.save_volume('models', self.cfg, True), '')
        self.assertIsNone(self.m.volume_controllers['models'].state['id'])
        with patch.object(self.m, 'backends', []):
            self.assertEqual(asyncio.run(self.m.delete_volume('models')), '')
        self.assertNotIn('models', self.m.volume_names())

    def test_delete_confirmed_gone_removes_config_and_state(self):
        """Confirmed external deletion must be removable without creating or deleting a billed volume."""
        self.m.save_volume('models', self.cfg, True)
        c = self.m.volume_controllers['models']
        c.state.update(id='gone', missing=2)
        self.m._volume_save_state('models', c.state)
        with patch.object(self.m, 'backends', []), patch.object(c, 'delete_volume', AsyncMock()) as delete:
            self.assertEqual(asyncio.run(self.m.delete_volume('models')), '')
            delete.assert_not_awaited()
        self.assertNotIn('models', self.m.volume_names())
        self.assertIsNone(self.m._volume_load_state('models'))

    def test_fetch_execution_budget_override(self):
        """Model transfers must pass their own byte-derived budget instead of a generation timeout."""
        ad = Mock(spec=self.m.adapters.RunpodAdapter, run_op=AsyncMock(return_value={}))
        b = dict(name='rp1', type='runpod', volume='models', max_wait=123)
        with patch.object(self.m, 'backends', [b]), patch.object(self.m, 'backend_adapters', {'runpod:rp1': ad}):
            d = self.m._volume_deps('models')
            on_id = Mock()
            asyncio.run(d.run_fetch('runpod:rp1', {'op': 'fetch'}, on_id, max_wait=1300))
            ad.run_op.assert_awaited_once_with({'op': 'fetch'}, 1300, on_id)
