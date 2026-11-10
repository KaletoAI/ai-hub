"""Pure wiro module rules: sample defaults are never sent, category mapping skips
realtime/llm rows, the option/default/client precedence, multipart slot expansion,
the task-detail state machine (task_end/task_error are NOT terminal, pexit decides),
and key redaction. Fixtures are small excerpts shaped like the public Tool/List
snapshot, inline — never a copy of the whole catalog.
"""
import hashlib
import hmac
import json
import unittest

import wiro


def _item(id, type, required=False, defaultvalue="", hidden=None, maxinputlenght=None,
          minvalue="", maxvalue="", options=None, note=""):
    item = {"id": id, "type": type, "required": required, "defaultvalue": defaultvalue,
            "value": "", "minvalue": minvalue, "maxvalue": maxvalue, "options": options or [],
            "note": note, "label": id}
    if hidden is not None:
        item["hidden"] = hidden
    if maxinputlenght is not None:
        item["maxinputlenght"] = maxinputlenght
    return item


# nano-banana-shaped image row (sample URL defaults on the file input — W6's trap).
IMAGE_ROW = {"cleanslugowner": "google", "cleanslugproject": "nano-banana",
             "title": "Nano Banana", "description": "image edit",
             "categories": ["tool", "text-to-image", "image-to-image", "image-editing"],
             "dynamicprice": json.dumps([{"inputs": {}, "price": 0.04, "priceMethod": "cpr"}]),
             "parameters": [{"title": "", "subtitle": "", "items": [
                 _item("inputImage", "combinefileinput", maxinputlenght=3,
                       defaultvalue=["https://cdn.wiro.ai/uploads/sampleinputs/a.png",
                                     "https://cdn.wiro.ai/uploads/sampleinputs/b.png"]),
                 _item("prompt", "textarea", required=True,
                       defaultvalue="make it sunny (wiro sample prompt)"),
                 _item("aspectRatio", "select", defaultvalue="square",
                       options=[{"text": "Square", "value": "square"},
                                {"text": "Wide", "value": "wide"}]),
                 _item("temperature", "float", defaultvalue="0.7", minvalue="0", maxvalue="5"),
                 _item("seed", "number", minvalue="1", maxvalue="100"),
                 _item("secretKnob", "text", hidden=True),
             ]}]}

# whisper-shaped ASR row: required file input with a SAMPLE mp3 default.
ASR_ROW = {"cleanslugowner": "openai", "cleanslugproject": "whisper",
           "title": "Whisper", "categories": ["tool", "speech-to-text"],
           "dynamicprice": json.dumps([{"inputs": {}, "price": 0.0006,
                                        "priceMethod": "cp-audiosecondslength"}]),
           "parameters": [{"title": "", "items": [
               _item("inputAudio", "combinefileinput", required=True,
                     defaultvalue="https://cdn.wiro.ai/uploads/sampleinputs/sample.mp3"),
               _item("language", "select", required=True, defaultvalue="en",
                     options=[{"text": "English", "value": "en"},
                              {"text": "Deutsch", "value": "de"}])]}]}

REALTIME_ROW = {"cleanslugowner": "google", "cleanslugproject": "gemini-3-8-live",
                "title": "Gemini Live",
                "categories": ["tool", "voice-agent", "speech-to-speech", "realtime-conversation"],
                "parameters": [{"title": "", "items": [_item("voice", "select", required=True)]}]}
LLM_ROW = {"cleanslugowner": "moonshot", "cleanslugproject": "kimi-k3-1-pro",
           "title": "Kimi", "categories": ["tool", "partner-llm", "llm-reasoning", "llm"],
           "parameters": [{"title": "", "items": [_item("prompt", "textarea", required=True)]}]}
VIDEO_ROW = {"cleanslugowner": "klingai", "cleanslugproject": "kling-v3",
             "title": "Kling", "categories": ["tool", "image-to-video", "text-to-video"],
             "dynamicprice": [{"inputs": {}, "price": 0.5, "priceMethod": "cpo"}],
             "parameters": [{"title": "", "items": [
                 _item("lastFrameImage", "fileinput"),
                 _item("inputVideo", "fileinput",
                       defaultvalue="https://cdn.wiro.ai/uploads/sampleinputs/in.mp4"),
                 _item("prompt", "text", required=True)]}]}


def _cand(row, endpoint="image", model=None, options=None):
    meta = {"name": row["title"], "parameters": row["parameters"],
            "dynamicprice": row.get("dynamicprice")}
    cand = wiro.candidate_for("acc", model or wiro.model_id_of(row), endpoint, meta)
    if options:
        cand["wiro"]["options"].update(options)
        wiro.options_of(cand)                   # pins are validated like candidate_for does
    return cand


class CatalogMapping(unittest.TestCase):
    def test_realtime_and_llm_rows_are_never_media_candidates(self):
        """Guards the W3 skips: realtime models are rejected by Run and llm/chat rows
        belong to the LLM endpoint — either becoming a candidate would be a paid task
        that can never deliver."""
        self.assertIsNone(wiro.tool_metadata(REALTIME_ROW))
        self.assertIsNone(wiro.tool_metadata(LLM_ROW))
        self.assertEqual(wiro.catalog_endpoint(["speech-to-speech", "realtime-conversation"]), None)
        self.assertEqual(wiro.catalog_endpoint(["speech-to-speech"]), "audio")

    def test_category_first_match_and_uncategorized_skip(self):
        """Guards W3's mapping order and the skip of rows with no media category."""
        self.assertEqual(wiro.catalog_endpoint(["text-to-speech"]), "speech")
        self.assertEqual(wiro.catalog_endpoint(["music-generation"]), "audio")
        self.assertEqual(wiro.catalog_endpoint(["image-to-song"]), "audio")
        self.assertEqual(wiro.catalog_endpoint(["talking-head"]), "video")
        self.assertEqual(wiro.catalog_endpoint(["upscale"]), "image")
        self.assertEqual(wiro.catalog_endpoint(["background-removal"]), "image")
        self.assertIsNone(wiro.catalog_endpoint(["chat", "image-to-text", "video-to-text"]))
        self.assertIsNone(wiro.catalog_endpoint([]))
        self.assertIsNone(wiro.catalog_endpoint("text-to-image"))

    def test_backend_owner_project_id_uses_clean_slugs(self):
        """Guards the `backend/owner/project` route id: slugproject may contain
        spaces, so the id is built from the CLEAN slugs and splits at the first /."""
        row = dict(VIDEO_ROW, slugproject="kling v3 fast")
        self.assertEqual(wiro.model_id_of(row), "klingai/kling-v3")
        entry = wiro.tool_metadata(row)
        self.assertEqual(entry[:2], ("klingai/kling-v3", "video"))
        bad = dict(VIDEO_ROW, cleanslugproject="")
        self.assertIsNone(wiro.tool_metadata(bad))

    def test_dynamicprice_string_and_list_both_parse(self):
        """Guards the Tool/List string / Detail array duality of dynamicprice."""
        self.assertEqual(len(wiro.dynamic_prices({"dynamicprice": IMAGE_ROW["dynamicprice"]})), 1)
        self.assertEqual(len(wiro.dynamic_prices({"dynamicprice": VIDEO_ROW["dynamicprice"]})), 1)
        self.assertEqual(wiro.dynamic_prices({"dynamicprice": "not json"}), [])
        self.assertEqual(wiro.dynamic_prices({"dynamicprice": None}), [])
        self.assertEqual(wiro.price_from({"dynamicprice": IMAGE_ROW["dynamicprice"]}), {"usd": 0.04})
        self.assertEqual(wiro.price_from({"dynamicprice": VIDEO_ROW["dynamicprice"]}), {"usd": 0.5})
        # cp-readoutput / per-second prices have no request-time USD: unknown, never 0.
        self.assertIsNone(wiro.price_from({"dynamicprice": ASR_ROW["dynamicprice"]}))


class AuthAndRedaction(unittest.TestCase):
    def test_signature_headers_hmac_key_message_secret_nonce(self):
        """Guards W1: hex HMAC_SHA256(key=KEY, msg=SECRET+NONCE) with a fresh nonce
        per request, x-api-key always the bare KEY."""
        headers = wiro.signature_headers("KEY:SECRET", now=1000)
        self.assertEqual(headers["x-api-key"], "KEY")
        nonce = headers["x-nonce"]
        self.assertTrue(nonce.isdigit())                      # docs: a timestamp or an integer
        self.assertEqual(int(nonce) // 1000, 1000)
        expect = hmac.new(b"KEY", ("SECRET" + nonce).encode(), hashlib.sha256).hexdigest()
        self.assertEqual(headers["x-signature"], expect)
        again = wiro.signature_headers("KEY:SECRET", now=1000)
        self.assertNotEqual(again["x-nonce"], nonce)          # same second, still unique
        self.assertEqual(again["x-signature"],
                         hmac.new(b"KEY", ("SECRET" + again["x-nonce"]).encode(),
                                  hashlib.sha256).hexdigest())

    def test_key_only_mode_has_no_signature_headers(self):
        """Guards the key-only project mode: only x-api-key, never a signature."""
        headers = wiro.signature_headers("KEY", now=1000)
        self.assertEqual(headers, {"x-api-key": "KEY"})
        self.assertEqual(wiro.split_key("KEY:SECRET:EXTRA"), ("KEY", "SECRET:EXTRA"))
        self.assertEqual(wiro.signature_headers(""), {})

    def test_both_key_halves_are_redacted_from_errors(self):
        """Guards that neither KEY nor SECRET nor the joined credential survives in
        an error/log text."""
        text = wiro.redact("key KEY secret SECRET joint KEY:SECRET", "KEY:SECRET")
        self.assertNotIn("KEY", text.replace("[redacted]", ""))
        self.assertNotIn("SECRET", text.replace("[redacted]", ""))
        self.assertEqual(wiro.redact("plain", ""), "plain")


class ValuePrecedence(unittest.TestCase):
    def test_file_and_text_defaults_are_never_sent(self):
        """Guards W6's silent trap: file-input defaults are wiro SAMPLE URLs and
        text/textarea defaults are sample prompts — filling either would generate
        from the vendor's samples instead of the client's request."""
        cand = _cand(IMAGE_ROW)
        body, parts = wiro.build_request(cand, {"prompt": "my own prompt"}, {})
        self.assertIsNone(parts)
        self.assertEqual(body["prompt"], "my own prompt")
        self.assertNotIn("inputImage", body)
        self.assertNotIn("seed", body)
        # Select/number/float/checkbox defaults DO fill (W6's allow-list).
        self.assertEqual(body["aspectRatio"], "square")
        self.assertEqual(body["temperature"], 0.7)

    def test_required_text_without_value_is_input_error(self):
        """Guards the 400 before any paid POST: a required text/textarea without a
        client value is an INPUT_ERROR, never the sample default."""
        cand = _cand(IMAGE_ROW)
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.build_request(cand, {}, {})

    def test_required_file_without_upload_is_input_error(self):
        """Guards the same refusal for a required file input (whisper's inputAudio):
        the sample mp3 default must never stand in for the client's audio."""
        cand = _cand(ASR_ROW, endpoint="asr")
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.build_request(cand, {}, {})

    def test_unknown_client_key_is_input_error(self):
        """Guards the 400 for keys that are not parameters of the model: a typo must
        not bill a run with a silently dropped control."""
        cand = _cand(IMAGE_ROW)
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.build_request(cand, {"prompt": "x", "widthh": 512}, {})

    def test_options_json_pin_beats_catalog_default_client_beats_pin(self):
        """Guards the precedence client > alias option pin > wiro defaultvalue."""
        cand = _cand(IMAGE_ROW, options={"options_json": '{"aspectRatio": "wide"}'})
        body, _ = wiro.build_request(cand, {"prompt": "x"}, {})
        self.assertEqual(body["aspectRatio"], "wide")
        body, _ = wiro.build_request(cand, {"prompt": "x", "aspectRatio": "square"}, {})
        self.assertEqual(body["aspectRatio"], "square")
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.options_of({"wiro": {"options": {"options_json": '{"model": "x"}'}}})

    def test_range_and_choice_validation(self):
        """Guards minvalue/maxvalue and select-option checks before billing."""
        cand = _cand(IMAGE_ROW)
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.build_request(cand, {"prompt": "x", "temperature": 9}, {})
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.build_request(cand, {"prompt": "x", "aspectRatio": "tall"}, {})
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.build_request(cand, {"prompt": "x", "seed": 1.5}, {})

    def test_n_rules(self):
        """Guards W6's n handling: n>1 without a samples/numberOfImages param is an
        INPUT_ERROR; n==1 is ignored."""
        cand = _cand(IMAGE_ROW)
        body, _ = wiro.build_request(cand, {"prompt": "x", "n": 1}, {})
        self.assertNotIn("n", body)
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.validate_request(cand, {"prompt": "x", "n": 2}, (), ())
        row = dict(IMAGE_ROW)
        row["parameters"] = [{"title": "", "items": IMAGE_ROW["parameters"][0]["items"] + [
            _item("samples", "number", minvalue="1", maxvalue="4")]}]
        cand = _cand(row)
        wiro.validate_request(cand, {"prompt": "x", "n": 2, "samples": 2}, (), ())


class Uploads(unittest.TestCase):
    def test_hidden_items_never_become_public_fields(self):
        """Guards W5: hidden parameter items stay out of params, images and files —
        a control the vendor hides must not be settable through the gateway."""
        params, images, files = wiro.public_fields(_cand(IMAGE_ROW))
        names = {f["name"] for f in params} | {f["name"] for f in images} | {f["name"] for f in files}
        self.assertNotIn("secretKnob", names)
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.build_request(_cand(IMAGE_ROW), {"prompt": "x", "secretKnob": "1"}, {})

    def test_image_slots_expand_and_repeat_in_multipart(self):
        """Guards W5/W6's expansion: maxinputlenght 3 → inputImage, inputImage_2,
        inputImage_3 slots, all sent as REPEATED inputImage fields in slot order,
        with no sample default anywhere."""
        cand = _cand(IMAGE_ROW)
        self.assertEqual(wiro.slot_names(cand), ["inputImage", "inputImage_2", "inputImage_3"])
        body, parts = wiro.build_request(cand, {"prompt": "x"},
            {"inputImage": b"one", "inputImage_2": b"two", "inputImage_3": b"three"})
        self.assertEqual([(fid, data) for fid, _name, data in parts],
                         [("inputImage", b"one"), ("inputImage", b"two"), ("inputImage", b"three")])
        self.assertNotIn("inputImage", body)

    def test_fileinput_sends_file_plus_empty_url_field(self):
        """Guards the fileinput wire shape: the file under its id AND an empty
        {id}Url, per the task-API contract."""
        cand = _cand(VIDEO_ROW, endpoint="video")
        body, parts = wiro.build_request(cand, {"prompt": "x"}, {"lastFrameImage": b"img"})
        self.assertIn(("lastFrameImage", "lastFrameImage.bin", b"img"), parts)
        self.assertIn(("lastFrameImageUrl", None, b""), parts)

    def test_last_frame_and_document_ids_are_file_slots(self):
        """Guards W5's slot classification: ids containing `image` (or exactly
        lastFrameImage) are IMAGE slots; audio/video/document ids are FILE fields
        with a derived accept list."""
        _params, images, files = wiro.public_fields(_cand(VIDEO_ROW, endpoint="video"))
        self.assertIn("lastFrameImage", {f["name"] for f in images})
        by_name = {f["name"]: f for f in files}
        self.assertEqual(by_name["inputVideo"]["accept"], ["mp4", "webm", "mov"])
        _p, _i, asr_files = wiro.public_fields(_cand(ASR_ROW, endpoint="asr"))
        self.assertEqual({f["name"] for f in asr_files}, {"inputAudio"})
        self.assertTrue(asr_files[0]["required"])

    def test_asr_upload_flow(self):
        """Guards that the required whisper audio arrives as multipart, not JSON."""
        cand = _cand(ASR_ROW, endpoint="asr")
        body, parts = wiro.build_request(cand, {}, {}, {"inputAudio": ("clip.mp3", b"audio")})
        self.assertEqual(body["language"], "en")             # select default fills
        self.assertEqual([(fid, fn, data) for fid, fn, data in parts],
                         [("inputAudio", "clip.mp3", b"audio")])

    def test_request_summary_carries_no_file_bytes(self):
        """Guards that the job row's request summary names byte counts, never bytes."""
        body, parts = wiro.build_request(_cand(VIDEO_ROW, endpoint="video"),
            {"prompt": "x"}, {"lastFrameImage": b"img"})
        summary = wiro.request_summary(body, parts)
        self.assertNotIn("img", json.dumps(summary))
        self.assertIn("<files>", summary)


class TaskStates(unittest.TestCase):
    def test_task_end_and_task_error_are_not_terminal(self):
        """Guards W8's state machine: task_end comes BEFORE post-processing (outputs
        not final) and task_error/task_error_full are transient — treating either as
        a verdict would discard a paid, still-running task."""
        for status in ("task_end", "task_error", "task_error_full", "task_queue",
                       "task_output", "task_postprocess_start"):
            state = wiro.parse_task({"status": status})
            self.assertIsNone(state.error, status)
            self.assertEqual(state.status, status.upper())

    def test_pexit_nonzero_fails_the_finished_task(self):
        """Guards that task_postprocess_end alone is NOT success: pexit != "0"
        (numbers are strings) fails with the redactable debugerror."""
        state = wiro.parse_task({"status": "task_postprocess_end", "pexit": "1",
                                 "debugerror": "OOM in the worker", "outputs": []})
        self.assertEqual(state.error, "OOM in the worker")
        state = wiro.parse_task({"status": "task_cancel"})
        self.assertEqual(state.error, "task canceled")

    def test_success_needs_terminal_pexit_zero_and_outputs(self):
        """Guards W8's success test: terminal + pexit "0" + ≥1 output; a finished
        task without outputs is an error, never a silent success."""
        state = wiro.parse_task({"status": "task_postprocess_end", "pexit": "0",
                                 "outputs": [], "totalcost": "0.04"})
        self.assertEqual(state.error, "wiro task finished without outputs")
        state = wiro.parse_task({"status": "task_postprocess_end", "pexit": "0",
                                 "totalcost": "0.04", "elapsedseconds": "12",
                                 "outputs": [{"name": "out.mp4", "contenttype": "video/mp4",
                                              "size": "100", "url": "https://cdn.test/out.mp4"}]})
        self.assertIsNone(state.error)
        self.assertEqual(state.downloads, [("out.mp4", "https://cdn.test/out.mp4")])
        self.assertEqual(state.credits, 0.04)
        self.assertEqual(state.progress, 100)

    def test_unknown_status_is_a_verdict(self):
        """Guards that a status outside the documented list is final with an
        explanation, never polled to max_wait holding the in-flight slot."""
        state = wiro.parse_task({"status": "task_explode"})
        self.assertIn("unknown wiro task status", state.error)
        state = wiro.parse_task("not a dict")
        self.assertIsNotNone(state.error)

    def test_task_of_detail_envelope(self):
        """Guards the tasklist[0] unwrap and the refusal of a result:false body."""
        task = wiro.task_of_detail({"result": True, "tasklist": [{"status": "task_start"}]})
        self.assertEqual(task["status"], "task_start")
        with self.assertRaises(wiro.WiroInput):
            wiro.task_of_detail({"result": False, "errors": [{"code": 1, "message": "gone"}]})
        with self.assertRaises(wiro.WiroInput):
            wiro.task_of_detail({"result": True, "tasklist": []})

    def test_run_created_envelope(self):
        """Guards W7: result:true + taskid + socketaccesstoken = created; a malformed
        success is a final error, a refusal defers to the error-code classification."""
        self.assertEqual(wiro.run_created({"result": True, "taskid": "2221",
                                           "socketaccesstoken": "tok"}), ("2221", "tok"))
        self.assertIsNone(wiro.run_created({"result": False, "errors": [
            {"code": 96, "message": "concurrency"}]}))
        self.assertEqual(wiro.error_code({"result": False, "errors": [{"code": 96}]}), 96)
        self.assertIsNone(wiro.error_code({"result": True}))
        with self.assertRaises(wiro.WiroInput):
            wiro.run_created({"result": True, "taskid": "2221"})
        with self.assertRaises(wiro.WiroInput):
            wiro.run_created(["not a dict"])


if __name__ == "__main__":
    unittest.main()


class VendorDefaults(unittest.TestCase):
    def test_control_defaults_go_out_verbatim_prose_and_files_never(self):
        """Guards _vendor_default: wiro's catalog spells some control defaults outside
        their own option list ('English' vs 'en', ['vocal'], checkbox '--use_pe') —
        validating them would refuse every run of those models, so they are sent
        verbatim; a numeric `text` default (seed '0') is a value and goes out too,
        while a prose `text` default (a sample prompt) never does."""
        row = {"parameters": [{"items": [
            {"id": "prompt", "type": "textarea", "required": True, "defaultvalue": "sample"},
            {"id": "language", "type": "select", "defaultvalue": "English",
             "options": [{"text": "English", "value": "en"}]},
            {"id": "mode", "type": "select", "defaultvalue": ["vocal"],
             "options": [{"text": "vocal", "value": "vocal"}]},
            {"id": "use_pe", "type": "checkbox", "defaultvalue": "--use_pe"},
            {"id": "genre", "type": "multiselect", "defaultvalue": ["Pop"],
             "options": [{"id": "Pop", "name": "Pop"}, {"id": "rock", "name": "rock"}]},
            {"id": "seed", "type": "text", "required": True, "defaultvalue": "0"},
            {"id": "style", "type": "text", "defaultvalue": "cinematic sample"},
        ]}]}
        cand = wiro.candidate_for("w", "o/p", "audio", row)
        body, parts = wiro.build_request(cand, {"prompt": "x"}, {}, {})
        self.assertIsNone(parts)
        self.assertEqual(body, {"prompt": "x", "language": "English", "mode": ["vocal"],
                                "use_pe": "--use_pe", "genre": ["Pop"], "seed": "0"})
        # a CLIENT value is still validated against the options (value or id spelling)
        wiro.build_request(cand, {"prompt": "x", "genre": "rock", "language": "en"}, {}, {})
        with self.assertRaises(wiro.INPUT_ERROR):
            wiro.build_request(cand, {"prompt": "x", "genre": "jazz"}, {}, {})
        wiro.public_fields(cand)                 # odd defaults never break the schema


class OpenAIShimKeys(unittest.TestCase):
    ROW = {"parameters": [{"items": [
        {"id": "prompt", "type": "textarea", "required": True},
        {"id": "aspectRatio", "type": "select", "defaultvalue": "1:1",
         "options": [{"text": "1:1", "value": "1:1"}, {"text": "16:9", "value": "16:9"}]},
        {"id": "samples", "type": "number", "defaultvalue": "1", "minvalue": "1", "maxvalue": "4"}]}]}

    def test_openai_generic_keys_translate_or_drop_never_400(self):
        """The OpenAI Images shim passes size/quality/style/… for EVERY client (Open WebUI
        sends size=1024x1024): refusing them as unknown made those clients a 400 on most
        wiro models. size → the model's aspect ratio (or width/height), the rest dropped;
        n → the model's own sample count."""
        cand = wiro.candidate_for("w", "o/p", "image", self.ROW)
        body, _ = wiro.build_request(cand, {"prompt": "x", "size": "1920x1080", "quality": "hd",
                                            "style": "vivid", "n": 3}, {}, {})
        self.assertEqual(body, {"prompt": "x", "aspectRatio": "16:9", "samples": 3})
        body, _ = wiro.build_request(cand, {"prompt": "x", "size": "1000x700"}, {}, {})
        self.assertEqual(body["aspectRatio"], "1:1")          # no matching ratio: model default
        sized = {"parameters": [{"items": [{"id": "prompt", "type": "textarea"},
                                           {"id": "width", "type": "number"},
                                           {"id": "height", "type": "number"}]}]}
        body, _ = wiro.build_request(wiro.candidate_for("w", "o/p", "image", sized),
                                     {"prompt": "x", "size": "768x512"}, {}, {})
        self.assertEqual((body["width"], body["height"]), (768, 512))

    def test_text_outputs_become_a_transcript(self):
        """wiro speech-to-text answers in the task body, not as a file: a finished asr
        task with text but no output URL is a success, not "finished without outputs"."""
        state = wiro.parse_task({"status": "task_postprocess_end", "pexit": "0", "outputs": [
            {"contenttype": "raw", "content": {"raw": "hello world"}}]}, endpoint="asr")
        self.assertIsNone(state.error)
        self.assertEqual(state.texts, ["hello world"])
        state = wiro.parse_task({"status": "task_postprocess_end", "pexit": "0", "outputs": [],
                                 "debugoutput": "plain transcript"}, endpoint="asr")
        self.assertEqual(state.texts, ["plain transcript"])

    def test_image_upload_is_named_by_its_bytes(self):
        """An image slot carries bytes only; a `.bin` octet-stream upload risks the
        vendor refusing or misreading the file after the run is billed."""
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        row = {"parameters": [{"items": [{"id": "prompt", "type": "textarea"},
                                         {"id": "inputImage", "type": "fileinput", "required": True}]}]}
        _, parts = wiro.build_request(wiro.candidate_for("w", "o/p", "image", row),
                                      {"prompt": "x"}, {"inputImage": buf.getvalue()}, {})
        self.assertEqual(parts[0][:2], ("inputImage", "inputImage.png"))
        self.assertEqual(parts[1], ("inputImageUrl", None, b""))
