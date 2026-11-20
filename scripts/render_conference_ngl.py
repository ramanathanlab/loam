#!/usr/bin/env python3
"""Render the bundled offline NGL page with a temporary Playwright browser."""
import argparse,base64,json,time
from pathlib import Path
from playwright.sync_api import sync_playwright

def main(out, transparent_only=False):
    images=out/'ngl-images';images.mkdir(exist_ok=True);errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1920,'height':1080},device_scale_factor=1)
        page.on('pageerror',lambda error:(errors.append(str(error)),print('PAGEERROR',str(error),flush=True)))
        page.goto((out/'ngl-gallery.html').as_uri(),wait_until='load',timeout=120000)
        page.wait_for_function('window.renderReady===true || (window.errors && window.errors.length>0)',timeout=180000)
        assert not errors and not page.evaluate('window.errors'),errors
        details=page.evaluate('''() => ({nglVersion:NGL.Version,models:components.map(o=>({atoms:o.structure.atomCount,residues:o.structure.residueStore.count})),webgl:stages[0].viewer.renderer.getContext().getParameter(stages[0].viewer.renderer.getContext().VERSION)})''')
        print(json.dumps(details),flush=True)
        for view in ['site','context']:
            page.evaluate('(v)=>setView(v)',view)
            for style in ['clean','interface']:
                page.evaluate('(s)=>setStyle(s)',style)
                for i in range(3):
                    url=page.evaluate('(i)=>imageData(i,'+str(transparent_only).lower()+')',i);suffix='-transparent' if transparent_only else '';path=images/f'binder-{i+1}-{style}-{view}{suffix}.png';path.write_bytes(base64.b64decode(url.split(',',1)[1]));print(str(path),flush=True)
        page.evaluate('()=>setView("site")');page.screenshot(path=str(out/'ngl-gallery-preview.png'),full_page=True)
        assert not errors,errors
        record={'rendered_at':time.time(),'renderer':'NGL2.4.0 in Chromium '+browser.version,'graphics':'CPU SwiftShader WebGL; no compute allocation','details':details,'page_errors':errors,'images':12,'transparent_background':transparent_only}
        (out/('ngl-transparent-render-verification.json' if transparent_only else 'ngl-render-verification.json')).write_text(json.dumps(record,indent=2)+'\n');browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--transparent-only',action='store_true');a=p.parse_args();main(a.output.resolve(),a.transparent_only)
