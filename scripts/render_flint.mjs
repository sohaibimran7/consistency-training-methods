#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

import { assembleVegaLite } from 'flint-chart';
import * as vega from 'vega';
import { compile } from 'vega-lite';

function parseArgs(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!flag?.startsWith('--') || value === undefined) {
      throw new Error('usage: node scripts/render_flint.mjs --data DATA.json --spec CHART.json --output FIGURE.svg');
    }
    values[flag.slice(2)] = value;
  }
  for (const required of ['data', 'spec', 'output']) {
    if (!values[required]) throw new Error(`missing --${required}`);
  }
  return values;
}

function readJson(file, label) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    throw new Error(`cannot read ${label} JSON ${file}: ${error.message}`);
  }
}

const args = parseArgs(process.argv.slice(2));
if (fs.existsSync(args.output)) throw new Error(`refusing to overwrite existing output: ${args.output}`);
const compiledOutput = `${args.output}.vl.json`;
if (fs.existsSync(compiledOutput)) throw new Error(`refusing to overwrite existing output: ${compiledOutput}`);

const data = readJson(args.data, 'data');
if (!Array.isArray(data) || data.length === 0) throw new Error('data JSON must be a non-empty array');
const chart = readJson(args.spec, 'Flint specification');
if (!chart.semantic_types || !chart.chart_spec) {
  throw new Error('Flint specification must contain semantic_types and chart_spec');
}

const vegaLiteSpec = assembleVegaLite({ ...chart, data: { values: data } });
const vegaSpec = compile(vegaLiteSpec).spec;
const view = new vega.View(vega.parse(vegaSpec), { renderer: 'none' });
const svg = await view.toSVG();

fs.mkdirSync(path.dirname(args.output), { recursive: true });
fs.writeFileSync(compiledOutput, `${JSON.stringify(vegaLiteSpec, null, 2)}\n`);
fs.writeFileSync(args.output, `${svg}\n`);
console.log(`Wrote Flint chart to ${args.output}`);
console.log(`Wrote compiled Vega-Lite specification to ${compiledOutput}`);
