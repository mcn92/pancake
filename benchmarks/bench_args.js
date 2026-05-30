'use strict';

function parseInteger(value, flagName) {
  const parsed = parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Invalid value for ${flagName}: ${value}`);
  }
  return parsed;
}

function parseIntegerList(value, flagName) {
  const raw = String(value).split(',');
  const values = raw
    .map(part => part.trim())
    .filter(Boolean)
    .map(part => parseInteger(part, flagName));
  if (values.length === 0) {
    throw new Error(`Expected at least one value for ${flagName}`);
  }
  return values;
}

function readEnvInt(name) {
  if (!(name in process.env) || process.env[name] === '') return undefined;
  return parseInteger(process.env[name], name);
}

function readEnvIntList(name) {
  if (!(name in process.env) || process.env[name] === '') return undefined;
  return parseIntegerList(process.env[name], name);
}

function parseBenchmarkArgs(argv = process.argv.slice(2)) {
  const remaining = [];
  let m;
  let efConstruction;
  let efSearch;
  let efSearchValues;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--m') {
      if (i + 1 >= argv.length) throw new Error('Missing value for --m');
      m = parseInteger(argv[++i], '--m');
      continue;
    }
    if (arg === '--ef-construction') {
      if (i + 1 >= argv.length) throw new Error('Missing value for --ef-construction');
      efConstruction = parseInteger(argv[++i], '--ef-construction');
      continue;
    }
    if (arg === '--ef-search') {
      if (i + 1 >= argv.length) throw new Error('Missing value for --ef-search');
      efSearch = parseInteger(argv[++i], '--ef-search');
      continue;
    }
    if (arg === '--ef-search-values') {
      if (i + 1 >= argv.length) throw new Error('Missing value for --ef-search-values');
      efSearchValues = parseIntegerList(argv[++i], '--ef-search-values');
      continue;
    }
    remaining.push(arg);
  }

  if (m === undefined) m = readEnvInt('PANCAKE_BENCH_M');
  if (efConstruction === undefined) efConstruction = readEnvInt('PANCAKE_BENCH_EF_CONSTRUCTION');
  if (efSearch === undefined) efSearch = readEnvInt('PANCAKE_BENCH_EF_SEARCH');
  if (efSearchValues === undefined) efSearchValues = readEnvIntList('PANCAKE_BENCH_EF_SEARCH_VALUES');

  if (efSearchValues === undefined && efSearch !== undefined) {
    efSearchValues = [efSearch];
  }

  return {
    args: remaining,
    m,
    efConstruction,
    efSearch,
    efSearchValues,
  };
}

function resolveSingleValue(override, fallback) {
  return override === undefined ? fallback : override;
}

function resolveSweepValues(parsedArgs, fallbackValues) {
  if (parsedArgs.efSearchValues !== undefined) return parsedArgs.efSearchValues;
  if (parsedArgs.efSearch !== undefined) return [parsedArgs.efSearch];
  return fallbackValues;
}

module.exports = {
  parseBenchmarkArgs,
  resolveSingleValue,
  resolveSweepValues,
};
