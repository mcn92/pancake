'use strict';

const assert = require('node:assert/strict');
const native = require('./index.js');

const handle = native.pancake_init(4, 8, 0, 0, 4, 16, 16);
assert.notEqual(handle, 0xFFFFFFFF);

try {
  assert.throws(
    () => native.pancake_add(handle, new Float32Array(3)),
    /shorter than the index dimension/
  );
  assert.throws(
    () => native.pancake_add(handle, new Uint8Array(4)),
    /must be a Float32Array/
  );
  assert.throws(
    () => native.pancake_bulk_insert(handle, new Float32Array(7), 2),
    /shorter than n \* dimension/
  );
  assert.throws(
    () => native.pancake_bulk_insert(handle, new Float32Array(8), -1),
    /n must be non-negative/
  );
  assert.throws(
    () => native.pancake_query(handle, new Float32Array(3), 1),
    /shorter than the index dimension/
  );
  assert.throws(
    () => native.pancake_query(handle, new Uint8Array(4), 1),
    /must be a Float32Array/
  );

  assert.equal(native.pancake_add(handle, new Float32Array([0, 0, 0, 0])), 0);
  const result = native.pancake_query(handle, new Float32Array([0, 0, 0, 0]), 99);
  assert.equal(result.count, 1);
} finally {
  native.pancake_dispose(handle);
}

console.log('Native validation checks passed.');
