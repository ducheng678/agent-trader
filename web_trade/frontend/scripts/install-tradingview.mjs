import { spawnSync } from 'node:child_process';
import { cp, mkdir, rm, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const target = resolve(root, 'public', 'charting_library');
const tempRepo = resolve(root, '.tmp-tradingview-charting-library');

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function resolveLibrarySource(sourceRoot) {
  const direct = resolve(sourceRoot);
  if (await exists(resolve(direct, 'charting_library.js'))) return direct;
  const nested = resolve(direct, 'charting_library');
  if (await exists(resolve(nested, 'charting_library.js'))) return nested;
  throw new Error(`No charting_library.js found in ${direct} or ${nested}`);
}

async function copyLibrary(sourceRoot) {
  const source = await resolveLibrarySource(sourceRoot);
  await mkdir(dirname(target), { recursive: true });
  await rm(target, { recursive: true, force: true });
  await cp(source, target, { recursive: true });
  console.log(`Installed TradingView charting_library from ${source}`);
  console.log(`Target: ${target}`);
}

async function installFromRepo(repoUrl) {
  await rm(tempRepo, { recursive: true, force: true });
  const clone = spawnSync('git', ['clone', '--depth=1', repoUrl, tempRepo], {
    cwd: root,
    stdio: 'inherit'
  });
  if (clone.status !== 0) {
    throw new Error(`git clone failed for ${repoUrl}`);
  }
  try {
    await copyLibrary(tempRepo);
  } finally {
    await rm(tempRepo, { recursive: true, force: true });
  }
}

const localPath = process.env.TRADINGVIEW_LIBRARY_PATH;
const repoUrl = process.env.TRADINGVIEW_LIBRARY_REPO;

if (process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log('Install official TradingView Advanced Charts files into public/charting_library/.');
  console.log('');
  console.log('Use one of:');
  console.log('  TRADINGVIEW_LIBRARY_PATH=/path/to/charting_library npm run install:tradingview');
  console.log('  TRADINGVIEW_LIBRARY_REPO=git@github.com:vendor/private-repo.git npm run install:tradingview');
  process.exit(0);
}

try {
  if (localPath) {
    await copyLibrary(localPath);
  } else if (repoUrl) {
    await installFromRepo(repoUrl);
  } else {
    throw new Error('Set TRADINGVIEW_LIBRARY_PATH to an official charting_library directory, or TRADINGVIEW_LIBRARY_REPO to the official private repo URL.');
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
