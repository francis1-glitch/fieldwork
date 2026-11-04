#!/usr/bin/env node
// Seals the Morning Brief so only the devices you turned on in Cockpit can read it.
// The repo is public: data/brief.json holds the date, an event count and ciphertext. Nothing readable.
//
//   node cockpit/tools/seal-brief.mjs <plain.json>      (run from the repo root)
//
// <plain.json> = {"date":"YYYY-MM-DD","urgency":{"score":6,"label":"Moderate"},
//                 "events":[{"time":"9:00 AM","title":"..."}],"priorities":["..."]}
// Keep <plain.json> OUTSIDE the repo. This script never writes plaintext into the repo.
// Device public keys live in data/cockpit-keys.json (public keys only; safe to publish).
// Scheme: a fresh AES-256-GCM key per brief, wrapped with each device's RSA-OAEP-2048 (SHA-256) key.
import fs from 'node:fs';
const { subtle } = globalThis.crypto;
const b64 = u => Buffer.from(u).toString('base64');
const unb64 = s => new Uint8Array(Buffer.from(s, 'base64'));
const scrub = s => String(s ?? '')
  .replace(/https?:\/\/\S+/gi, '')                       // meeting links
  .replace(/[\w.+-]+@[\w-]+\.[\w.-]+/g, '')               // emails
  .replace(/\+?\d[\d\s().-]{7,}\d/g, '')                  // phone numbers
  .replace(/\s{2,}/g, ' ').trim()
  .replace(/(\s+(at|via|on|with|@|:))+$/i, '').replace(/[\s\-—–:,;(]+$/, '')   // tidy what a removed link/number leaves behind
  .slice(0, 120);

const src = process.argv[2];
if (!src) { console.error('usage: node cockpit/tools/seal-brief.mjs <plain.json>'); process.exit(2); }
const p = JSON.parse(fs.readFileSync(src, 'utf8'));
if (!/^\d{4}-\d{2}-\d{2}$/.test(p.date || '')) { console.error('plain.json needs "date": "YYYY-MM-DD"'); process.exit(2); }
const clean = {
  date: p.date,
  urgency: p.urgency && Number.isFinite(+p.urgency.score) ? { score: Math.max(1, Math.min(10, Math.round(+p.urgency.score))), label: scrub(p.urgency.label).slice(0, 20) } : null,
  events: (Array.isArray(p.events) ? p.events : []).map(e => ({ time: scrub(e.time).slice(0, 12), title: scrub(e.title) })).filter(e => e.title).slice(0, 40),
  priorities: (Array.isArray(p.priorities) ? p.priorities : []).map(scrub).filter(Boolean).slice(0, 5),
};

let keys = [];
try { keys = (JSON.parse(fs.readFileSync('data/cockpit-keys.json', 'utf8')).keys || []).filter(k => k && k.kid && k.spki); } catch (e) {}

const out = { v: 2, date: clean.date, n: clean.events.length };
if (!keys.length) {
  out.locked = 'no-devices';   // nobody to seal for: publish nothing readable
} else {
  const aes = await subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt']);
  const raw = new Uint8Array(await subtle.exportKey('raw', aes));
  const iv = globalThis.crypto.getRandomValues(new Uint8Array(12));
  out.iv = b64(iv);
  out.ct = b64(new Uint8Array(await subtle.encrypt({ name: 'AES-GCM', iv }, aes, new TextEncoder().encode(JSON.stringify(clean)))));
  out.keys = [];
  for (const k of keys) {
    const pub = await subtle.importKey('spki', unb64(k.spki), { name: 'RSA-OAEP', hash: 'SHA-256' }, false, ['encrypt']);
    out.keys.push({ kid: k.kid, wk: b64(new Uint8Array(await subtle.encrypt({ name: 'RSA-OAEP' }, pub, raw))) });
  }
}
fs.mkdirSync('data', { recursive: true });
fs.writeFileSync('data/brief.json', JSON.stringify(out) + '\n');
console.log(`sealed brief for ${out.date}: ${out.n} events, ${keys.length} device key(s)` + (keys.length ? '' : ' — no devices yet, wrote a locked stub with no content'));
