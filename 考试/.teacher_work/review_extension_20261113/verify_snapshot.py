import os
import sys
import json
import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np

work = Path(__file__).resolve().parent
snapshot = work / 'submitted_snapshot'
os.chdir(snapshot)
checks = {}


def record(name, value):
    checks[name] = bool(value)
    print(name, bool(value))
    assert value, name


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


rotation = load('submitted_rotation', snapshot / 'rotate_image.py')
augmentation = load('submitted_augmentation', snapshot / 'augment_image.py')
process = load('submitted_process', snapshot / 'process_one.py')
img = decode(snapshot / 'input/input.png')
original = img.copy()
h, w = img.shape[:2]
matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 12, 1.0)
expected_rotation = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
for name, function in [('rotate_file', rotation.fixed_rotate),
                       ('augment_file_rotate', augmentation.fixed_rotate),
                       ('process_file_rotate', process.rotate_image)]:
    actual = function(img, 12)
    record(name + '_pixels', np.array_equal(actual, expected_rotation))
    record(name + '_shape_dtype', actual.shape == img.shape and actual.dtype == np.uint8)
    record(name + '_new_memory', not np.shares_memory(actual, img))
    record(name + '_input_unchanged', np.array_equal(img, original))

for name, function in [('augment_file', augmentation.random_aug),
                       ('process_file_augment', process.augment_image)]:
    rng = np.random.default_rng(21)
    replay = np.random.default_rng(21)
    seen = set()
    all_pixels_match = True
    for i in range(24):
        actual, flipped, angle = function(img, rng)
        expected_flip = replay.random() < 0.5
        expected_angle = float(replay.uniform(-15, 15))
        start = cv2.flip(img, 1) if expected_flip else img
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), expected_angle, 1.0)
        expected = cv2.warpAffine(start, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        all_pixels_match = all_pixels_match and np.array_equal(actual, expected)
        assert flipped == expected_flip and angle == expected_angle
        assert actual.shape == img.shape and actual.dtype == np.uint8
        assert not np.shares_memory(actual, img)
        seen.add(flipped)
    record(name + '_rng_sequence_and_pixels', all_pixels_match)
    record(name + '_both_flip_branches', seen == {False, True})
    record(name + '_input_unchanged', np.array_equal(img, original))

for mode in ['existing', 'clean_a', 'clean_b', 'cross_cwd']:
    folder = work / mode / 'output_supplement'
    names = sorted(p.name for p in folder.iterdir())
    record(mode + '_exact_names', names == ['input_flip.png', 'input_rotate.png'])
    flip = decode(folder / 'input_flip.png')
    rotate = decode(folder / 'input_rotate.png')
    record(mode + '_stored_shape_dtype', flip.shape == rotate.shape == (300, 400, 3)
           and flip.dtype == rotate.dtype == np.uint8)
    record(mode + '_flip_pixels', np.array_equal(flip, img[:, ::-1, :]))
    record(mode + '_rotation_pixels', np.array_equal(rotate, expected_rotation))
    record(mode + '_rotation_changed', not np.array_equal(rotate, img))

failure_folder = work / 'failure_tests'
failure_input = failure_folder / 'input'
failure_output = failure_folder / 'output'
failure_input.mkdir(parents=True, exist_ok=True)
failure_output.mkdir(parents=True, exist_ok=True)
(failure_input / 'corrupt.png').write_bytes(b'not an image')
(failure_input / 'zero.png').write_bytes(b'')
(failure_input / 'ignore.txt').write_text('ignored', encoding='utf-8')
(failure_input / 'nested.png').mkdir(exist_ok=True)
(failure_input / 'valid.png').write_bytes((snapshot / 'input/input.png').read_bytes())
record('unsupported_extension_rejected', process.process_one(str(failure_input), 'ignore.txt', str(failure_output))[0] is False)
record('corrupt_image_rejected', process.process_one(str(failure_input), 'corrupt.png', str(failure_output))[0] is False)
record('zero_byte_image_rejected', process.process_one(str(failure_input), 'zero.png', str(failure_output))[0] is False)
success, failure, messages = process.batch_augment(str(failure_input), str(failure_output))
print('mixed_batch', success, failure, messages)
record('mixed_batch_correct_counts', (success, failure) == (1, 2))
record('mixed_batch_exact_artifacts', sorted(p.name for p in failure_output.iterdir()) == ['valid_flip.png', 'valid_rotate.png'])
original_write = process.write_image
process.write_image = lambda path, image: False
status, message = process.process_one(str(failure_input), 'valid.png', str(failure_folder))
process.write_image = original_write
record('save_false_not_reported_as_success', status is False and '保存失败' in message)

before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in failure_output.iterdir()}
success, failure, messages = process.batch_augment(str(failure_input), str(failure_output))
after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in failure_output.iterdir()}
record('rerun_rejects_existing', (success, failure) == (0, 3))
record('rerun_hashes_unchanged', before == after)

source = str(snapshot / 'input/input.png')
print('opencv_absolute_unicode_read', cv2.imread(source) is not None)
print('opencv_relative_ascii_read', cv2.imread('input/input.png') is not None)
print('checks_passed', len(checks), 'of', len(checks))
(work / 'probe_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
