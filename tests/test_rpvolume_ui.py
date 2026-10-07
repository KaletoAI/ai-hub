"""Volume console contracts that can otherwise silently clear fields or bill twice."""
import asyncio
import unittest
from unittest.mock import patch, AsyncMock
from urllib.parse import urlencode

from starlette.requests import Request
import admin


def request(fields=None, query=''):
    body = urlencode(fields or {}, doseq=True).encode()
    async def receive():
        return {'type': 'http.request', 'body': body, 'more_body': False}
    return Request({'type': 'http', 'method': 'POST', 'path': '/',
                    'query_string': query.encode(), 'headers': []}, receive)


class VolumeUI(unittest.TestCase):
    def view(self, **kw):
        return dict(name='models', dc='EU-RO-1', id='vol-id', size_gb=50,
                    max_size_gb=200, used_bytes=10**9, phase='ready',
                    problems=['S3 key missing'], backends=[],
                    plan={'aliases': {'demo': {'need_bytes': 10**9, 'have_bytes': 0,
                                              'ready': False}},
                          'unknown': [['models/old', 100]]}, **kw)

    def test_card_renders_problems_and_table(self):
        """A ready volume must still offer unknown cleanup and display missing keys."""
        html = admin._volume_card('models', self.view())
        for text in ('S3 key missing', 'demo', '/ui/volumes/delete-unknown',
                     'name="vol"', 'vol-id', 'EU-RO-1'):
            self.assertIn(text, html)
        self.assertNotIn('<script', html)

    def test_delete_hidden_with_backends(self):
        """Referencing backends must hide irreversible volume deletion."""
        v = self.view()
        v['backends'] = ['worker']
        html = admin._volume_card('models', v)
        self.assertNotIn('/ui/volumes/delete?', html)
        self.assertIn('edit=runpod%3Aworker', html)

    def test_post_actions_listed(self):
        """Unlisted actions would become mutating GET links through _btn."""
        for action in ('save', 'sync', 'delete-unknown', 'delete'):
            self.assertIn('/ui/volumes/' + action, admin._POST_ACTIONS)

    def test_secret_rows_never_render_values(self):
        """Secret state may expose presence only, including the two S3 fields."""
        with patch.object(admin, '_provider_tokens', return_value={'runpod': True, 'runpod_s3': True}):
            html = admin._srv_keys_body({'eff': {}})
        for text in ('RunPod API key', 'RunPod S3 key', 'name="s3_id"', 'name="s3_secret"'):
            self.assertIn(text, html)
        self.assertNotIn('value="True"', html)

    def test_volume_select_rendered_in_all_panes(self):
        """Switching backend types must not clear an unrendered volume field."""
        with patch.object(admin, '_volume_names', return_value=['models'], create=True):
            for kind in ('openai', 'comfyui', 'runpod', 'meshy', 'anthropic'):
                html = admin._backend_form({'name': 'worker', 'type': kind, 'volume': 'models'}, [])
                self.assertIn('name="volume"', html)
                self.assertIn('value="models" selected', html)

    def test_missing_requires_billing_confirm(self):
        """Recreation must warn that an empty replacement starts a new bill."""
        html = admin._volume_card('models', self.view(missing=2))
        for text in ('Re-create volume', 'recreate=1', 'NEW, EMPTY', '50 GB', 'data-confirm='):
            self.assertIn(text, html)

    def test_save_refusal_preserves_typed_numbers(self):
        """Invalid numeric input must survive a refused save instead of becoming blank."""
        with patch.object(admin, '_save_volume', return_value='bad size', create=True), \
             patch.object(admin, '_backends_view', AsyncMock(return_value='refused')) as render:
            asyncio.run(admin.volume_save(request({'new': '1', 'name': 'models',
                        'datacenter': 'EU-RO-1', 'size_gb': 'oops', 'max_size_gb': ''})))
        self.assertEqual(render.call_args.kwargs['status'], 400)
        self.assertIn('value="oops"', render.call_args.kwargs['detail'])

    def test_delete_requires_typed_name(self):
        """A forged delete without the typed name must never call the controller."""
        with patch.object(admin, '_delete_volume', AsyncMock(), create=True) as delete, \
             patch.object(admin, '_backends_view', AsyncMock(return_value='refused')) as render:
            asyncio.run(admin.volume_delete(request({'vol': 'models', 'confirm_name': 'wrong'})))
        delete.assert_not_called()
        self.assertEqual(render.call_args.kwargs['status'], 400)

    def test_s3_join_keep_and_clear(self):
        """S3 credentials must save together; clear wins over typed fields."""
        with patch.object(admin, '_save_provider_token', return_value='') as save:
            asyncio.run(admin.server_provider_token(request({'provider': 'runpod_s3', 's3_id': 'user_x', 's3_secret': 'rps_y'})))
            save.assert_called_with('runpod_s3', 'user_x:rps_y')
            save.reset_mock()
            asyncio.run(admin.server_provider_token(request({'provider': 'runpod_s3'})))
            save.assert_not_called()
            asyncio.run(admin.server_provider_token(request({'provider': 'runpod_s3', 's3_id': 'user_x', 's3_secret': 'rps_y', 'api_key_clear': '1'})))
            save.assert_called_with('runpod_s3', '')

    def test_routes_are_post_only(self):
        """Every volume action must be registered without a mutating GET route."""
        from fastapi import FastAPI
        app = FastAPI()
        admin.register(app)
        routes = [r for r in app.routes if r.path.startswith('/ui/volumes/') and 'POST' in r.methods]
        self.assertEqual(len(routes), 4)
        for route in routes:
            self.assertEqual(route.methods, {'POST'})

    def test_edit_locks_dc_but_submits_value(self):
        """Disabling a created volume's DC must not silently clear it on save."""
        html = admin._volume_form(dict(self.view(), new=False))
        self.assertIn('<input readonly', html)
        self.assertIn('<select disabled', html)
        self.assertIn('type="hidden" name="datacenter" value="EU-RO-1"', html)

    def test_backend_refusal_keeps_volume(self):
        """An unknown volume must refuse before store writes and retain the selection."""
        with patch.object(admin.store, 'get_backend', return_value=None), \
             patch.object(admin.store, 'list_backends', return_value=[]), \
             patch.object(admin, '_gateway_info', return_value={'backends': []}), \
             patch.object(admin, '_volume_field_refusal', return_value='unknown RunPod volume') as validate, \
             patch.object(admin.store, 'upsert_backend') as save, \
             patch.object(admin, '_backends_view', AsyncMock(return_value='refused')) as render:
            asyncio.run(admin.backend_save(request({'name': 'worker', 'type': 'runpod',
                        'url': 'https://api.runpod.ai/v2/endpoint', 'volume': 'missing'})))
        validate.assert_called_once_with('missing')
        save.assert_not_called()
        self.assertEqual(render.call_args.kwargs['status'], 400)
        self.assertIn('value="missing" selected', render.call_args.kwargs['detail'])

    def test_transfer_visible_before_plan(self):
        """Restart recovery transfers must remain visible before a first plan exists."""
        v = self.view()
        v['plan'] = None
        v['transfers'] = [{'file': 'models/recover', 'bytes': 100, 'total': 200, 'via': 's3'}]
        self.assertIn('models/recover', admin._volume_card('models', v))

    def test_blank_datacenter_stays_blank_on_refusal(self):
        """A refused blank DC must not silently select a different billing location."""
        html = admin._volume_form({'new': True, 'name': 'models', 'datacenter': '',
                                   'size_gb': '50', 'max_size_gb': '200'}, 'unknown DC')
        self.assertIn('<option value="" selected>', html)

    def test_off_volume_page_keeps_polling(self):
        """Automatic first creation starts off; stopping polls there freezes its card."""
        v = self.view()
        v['phase'] = 'off'
        with patch.object(admin, '_volume_names', return_value=['models']), \
             patch.object(admin, '_volume_view', return_value=v), \
             patch.object(admin, '_gateway_info', return_value={'backends': []}), \
             patch.object(admin, '_host_names', return_value=[]), \
             patch.object(admin.store, 'is_active', return_value=False):
            response = asyncio.run(admin._backends_view({}))
        self.assertIn('data-live="3"', response.body.decode())

    def test_ready_volume_page_is_static(self):
        """A ready volume has nothing that moves: polling the Backends tab every 3 s
        for as long as any volume exists would morph it under the operator for good."""
        v = self.view()
        v.update(phase='ready', transfers=[], fetch_job=None)
        with patch.object(admin, '_volume_names', return_value=['models']), \
             patch.object(admin, '_volume_view', return_value=v), \
             patch.object(admin, '_gateway_info', return_value={'backends': []}), \
             patch.object(admin, '_host_names', return_value=[]), \
             patch.object(admin.store, 'is_active', return_value=False):
            response = asyncio.run(admin._backends_view({}))
        self.assertNotIn('data-live=', response.body.decode())

    def test_waiting_volume_page_is_static(self):
        """A blocked or offline volume must not keep morphing the Backends tab forever."""
        v = self.view()
        v.update(phase='waiting', transfers=[], fetch_job=None)
        with patch.object(admin, '_volume_names', return_value=['models']), \
             patch.object(admin, '_volume_view', return_value=v), \
             patch.object(admin, '_gateway_info', return_value={'backends': []}), \
             patch.object(admin, '_host_names', return_value=[]), \
             patch.object(admin.store, 'is_active', return_value=False):
            response = asyncio.run(admin._backends_view({}))
        self.assertNotIn('data-live=', response.body.decode())

    def test_part_leftover_has_cleanup_checkbox(self):
        """Worker partials must be visible and selectable alongside unknown volume objects."""
        v = self.view(leftovers=[['models/download.gw-part', 123]])
        v['phase'] = 'waiting'
        html = admin._volume_card('models', v)
        self.assertIn('name="path" value="models/download.gw-part"', html)
        self.assertIn('/ui/volumes/delete-unknown', html)
        self.assertEqual(v['plan']['unknown'], [['models/old', 100]])
