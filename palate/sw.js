// Palate service worker: when online, always load the newest app; offline, use the saved copy.
const CACHE="palate-v1-6-0";
const FILES=["./","./index.html","./manifest.json","./icon-192-v3.png","./icon-512-v3.png","./apple-touch-icon-v3.png"];
self.addEventListener("install",e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(FILES)));self.skipWaiting()});
self.addEventListener("activate",e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k.startsWith("palate-")&&k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()))});
self.addEventListener("fetch",e=>{
  if(e.request.method!=="GET")return;
  const u=new URL(e.request.url);
  if(u.origin!==location.origin)return;
  const isPage=e.request.mode==="navigate"||u.pathname.endsWith("/")||u.pathname.endsWith(".html");
  if(isPage){
    // network first, fall back to the saved copy when offline
    e.respondWith(fetch(e.request,{cache:"no-store"}).then(r=>{if(r&&r.ok){const cp=r.clone();caches.open(CACHE).then(c=>c.put(e.request,cp))}return r}).catch(()=>caches.match(e.request).then(h=>h||caches.match("./index.html"))));
    return;
  }
  e.respondWith(caches.match(e.request).then(hit=>{
    const net=fetch(e.request).then(r=>{if(r&&r.ok){const cp=r.clone();caches.open(CACHE).then(c=>c.put(e.request,cp))}return r}).catch(()=>hit);
    return hit||net;
  }));
});
