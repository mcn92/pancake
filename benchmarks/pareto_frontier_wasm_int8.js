#!/usr/bin/env node
'use strict';

/**
 * Focused DBpedia WASM u8/int8 Pareto benchmark.
 *
 * Defaults to:
 *   dataset=dbpedia, count=25000, M=12, ef_construction=75
 *   configs=pancake-wasm-u8,pancake-wasm-fp32,pancake-native-u8,pancake-native-fp32,
 *           usearch-wasm-int8,usearch-wasm-fp32,usearch-int8,usearch-fp32
 *
 * Extra CLI args are passed through and take precedence over these defaults.
 */

const { spawnSync } = require('child_process');
const path = require('path');

const passthrough = process.argv.slice(2);

function hasFlag(name) {
  return passthrough.includes('--' + name);
}

function addDefault(args, name, value) {
  if (!hasFlag(name)) args.push('--' + name, String(value));
}

const args = [...passthrough];
addDefault(args, 'dataset', 'dbpedia');
addDefault(args, 'count', '25000');
addDefault(args, 'configs', [
  'pancake-wasm-u8',
  'pancake-wasm-fp32',
  'pancake-native-u8',
  'pancake-native-fp32',
  'usearch-wasm-int8',
  'usearch-wasm-fp32',
  'usearch-int8',
  'usearch-fp32',
].join(','));
addDefault(args, 'm', '12');
addDefault(args, 'ef-construction', '75');

const script = path.join(__dirname, 'pareto_frontier.js');
const nodeArgs = process.execArgv.includes('--expose-gc') ? [] : ['--expose-gc'];
const result = spawnSync(process.execPath, [...nodeArgs, script, ...args], {
  cwd: path.join(__dirname, '..'),
  stdio: 'inherit',
});

if (result.error) throw result.error;
process.exit(result.status == null ? 1 : result.status);
