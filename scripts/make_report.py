"""Export self-contained offline HTML results with clearly separated synthetic/real evidence."""
from pathlib import Path
import json, base64, html
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'

def embed(path):
    p=R/path
    if not p.exists():return ''
    mime='image/png' if p.suffix=='.png' else 'image/jpeg'
    return f'data:{mime};base64,'+base64.b64encode(p.read_bytes()).decode()

def run():
    real=json.loads((R/'tum_real/real_mapping_report.json').read_text())
    ids=json.loads((R/'identity_benchmark/ablation.json').read_text())
    sem=json.loads((R/'semantic_ablation/ablations.json').read_text())
    move=json.loads((R/'dynamic_ablation/moved_objects.json').read_text())
    cmds=json.loads((R/'semantic_demo/operator_commands.json').read_text())['commands']
    replay=R/'tum_real/real_association_ablation.json'
    replaydata=json.loads(replay.read_text()) if replay.exists() else None
    css='''body{font-family:Inter,system-ui,sans-serif;background:#f6f7fb;color:#202938;margin:0}
      main{max-width:1150px;margin:38px auto;padding:0 25px} h1{font-size:32px}h2{margin-top:40px}
      section{background:white;border:1px solid #dce3ef;border-radius:16px;padding:24px;margin:18px 0}
      .pill{font-size:12px;padding:4px 9px;border-radius:20px;background:#e9edf4;display:inline-block}
      .cards{display:flex;gap:14px;flex-wrap:wrap}.card{flex:1;min-width:160px;background:#eef3fa;padding:16px;border-radius:12px}
      .stat{font-size:27px;font-weight:750}.sub{color:#607081;font-size:13px}img{width:100%;border-radius:9px}
      table{border-collapse:collapse;width:100%;}th,td{text-align:left;padding:10px;border-bottom:1px solid #e0e5ee}
      a{color:#3159a0} .warn{border-left:4px solid #e59737;padding:10px 14px;background:#fff7e5}
      .tabs{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.tabs button{border:1px solid #becbd9;background:#fff;padding:9px 13px;border-radius:8px;cursor:pointer}
      .tabs button:hover{background:#eaf2fc}.hidden{display:none}iframe{width:100%;height:650px;border:1px solid #d7dfe9;border-radius:9px}
      code{background:#eef1f8;padding:2px 5px;border-radius:4px}'''
    idrows=''.join(f"<tr><td>{html.escape(k)}</td><td>{v['idf1']:.5f}</td><td>{v['false_split_objects']}</td><td>{v['false_merge_tracks']}</td></tr>" for k,v in ids['results'].items())
    move_rows=''.join(f"<tr><td>{r['mode']}</td><td>{r['correct_reentry_after_swap']}/2</td></tr>" for r in move)
    cmd_rows=''.join(f"<tr><td>{html.escape(c['query'])}</td><td>{html.escape(c['status'])}</td><td>{html.escape(', '.join(x[:10] for x in c['target_ids']))}</td></tr>" for c in cmds)
    real_rows=''.join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k,v in sorted(real['labels'].items(),key=lambda k:-k[1])[:12])
    replayrows=(''.join(f"<tr><td>{html.escape(k)}</td><td>{v['canonical_tracks']}</td><td>{v['confirmed_tracks']}</td><td>{v['unresolved_observations']}</td><td>{v['candidate_duplicate_pairs']}</td></tr>" for k,v in replaydata.items()) if replaydata else '<tr><td colspan="5">Облачный replay-эксперимент ещё не завершён</td></tr>')
    txt=f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>RGB-D: объектная семантическая память — экспериментальный отчёт</title><style>{css}</style></head><body><main>
    <h1>Мультимодальная 3D-карта и постоянная объектная память</h1>
    <p>Исследовательский прототип, 28.10.2026. Разделение: реальные RGB-D данные, синтетические тесты, ещё не завершённые компоненты.</p>
    <section><h2>Реальная последовательность TUM Freiburg1 Desk</h2>
    <div class="cards"><div class="card"><div class="stat">{real['frames']}</div><div class="sub">RGB-D кадров с эталонными позами</div></div>
    <div class="card"><div class="stat">{real['detections']}</div><div class="sub">валидные 3D обнаружения</div></div>
    <div class="card"><div class="stat">{real['total_tracks']}</div><div class="sub">созданных треков</div></div>
    <div class="card"><div class="stat">{real['confirmed_tracks']}</div><div class="sub">подтвержденных треков</div></div>
    <div class="card"><div class="stat">{real['surface_voxels']}</div><div class="sub">3D вокселей поверхностей</div></div></div>
    <p class="warn"><b>Результат не демонстрирует отсутствие дублей:</b> несколько треков могут соответствовать одному предмету. TUM Desk не содержит эталонных instance-ID, поэтому IDF1 по реальным объектам неизвестен.</p>
    <img src="{embed(Path('tum_real/real_map_3d.png'))}" alt="Реальная карта"/>
    <p><a href="tum_real/real_map_interactive.html">Интерактивная 3D-карта</a> · <a href="tum_real/real_mapping_report.json">Полные метрики</a></p>
    <div class="tabs"> <button onclick="document.getElementById('overlay1').style.display='block';document.getElementById('overlay2').style.display='none'">Кадр RGB-D / детекция 1</button><button onclick="document.getElementById('overlay1').style.display='none';document.getElementById('overlay2').style.display='block'">Кадр RGB-D / детекция 2</button></div>
    <img id="overlay1" src="{embed(Path('tum_real/overlay_0300.jpg'))}" style="max-width:640px"/>
    <img id="overlay2" src="{embed(Path('tum_real/overlay_0480.jpg'))}" style="max-width:640px;display:none"/>
    <p class="sub">Контуры являются фактическими выводами предобученной YOLO11n-seg; обнаружены ложные классы.</p>
    <h3>Популярные метки детектора</h3><table><tr><th>Метка</th><th>Наблюдений</th></tr>{real_rows}</table></section>
    <section><h2>Повторное сопоставление реальных детекций</h2>
    <table><tr><th>Вариант</th><th>Созданные треки</th><th>Подтверждённые</th><th>Неопределённые наблюдения</th><th>Пары-кандидаты на дубли</th></tr>{replayrows}</table>
    <p class="sub">Число треков — НЕ достоверная метрика количества физических объектов; уменьшение может быть следствием ложного слияния.</p></section>
    <section><h2>Синтетическая карта и операторские команды</h2>
    <img src="{embed(Path('semantic_demo/map_2d.png'))}" alt="Синтетическая карта и маршруты"/>
    <p>Серые участки — препятствия; цветные линии — пути A*. Они не исполнялись на роботе.</p>
    <a href="semantic_demo/interactive_3d.html">Интерактивная синтетическая 3D-карта объектов</a>
    <table><tr><th>Команда</th><th>Реакция</th><th>Объекты</th></tr>{cmd_rows}</table></section>
    <section><h2>Идентичность объектов — синтетический тест</h2><p>12 объектов, 120 временных шагов, 622 наблюдения.</p>
    <table><tr><th>Алгоритм</th><th>IDF1</th><th>Физических объектов разделено</th><th>Ошибочных общих ID</th></tr>{idrows}</table>
    <h3>Два стула поменялись местами после исчезновения</h3><table><tr><th>Метод</th><th>Верные восстановленные ID</th></tr>{move_rows}</table></section>
    <section><h2>Воксельное голосование и выбор ракурсов (синтетика)</h2>
    <p>400 вокселей: первая шумная метка <b>{100*sem['single_view_accuracy']:.2f}%</b> правильных, после 5 голосов <b>{100*sem['five_view_vote_accuracy']:.2f}%</b> правильных.
    Потенциальные вызовы VLM: {sem['vlm_queries_without_selection']} → {sem['vlm_queries_after_selection']} (−{sem['reduction_vlm_queries_pct']}%).</p>
    <p class="warn">Это модель синтетического шума и выборки, без запуска реальной VLM.</p></section>
    <section><h2>Вывод</h2><p>Работает воспроизводимый CPU-конвейер реальной RGB-D сегментации, 3D проекции, сохранения объектных ID и визуализации. Однако уникальность ID пока не гарантирована: нужны повторная глобальная оптимизация, сильный appearance encoder, проверка occupancy и тестирование на сценах с annotated instance IDs. Реальные перемещения и Nav2-исполнение не проверены.</p>
    <p>Рекомендуемая исследовательская архитектура: RGB-D + версии поз RTAB-Map → class-agnostic 3D proposals → appearance+geometry fusion с неопределённостью → persistent instances → voxel-semantic evidence → scene graph → selective VLM → NL goal → Nav2 с безопасностью и проверкой доступности.</p>
    <p><a href="https://github.com/KiRiII-C/cv_vlm/tree/feature/semantic-memory-mapping">GitHub: исходный код</a> · <a href="https://github.com/KiRiII-C/cv_vlm/actions/runs/40700640646">GitHub: реальный запуск</a></p></section></main></body></html>'''
    path=R/'project_report.html';path.write_text(txt,encoding='utf-8')
    print('REPORT',path,path.stat().st_size)
if __name__=='__main__':run()
