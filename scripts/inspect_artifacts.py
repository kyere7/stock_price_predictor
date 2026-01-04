import os, joblib, sys

# Ensure project root is on sys.path so local model classes (e.g., HCLA_model)
# referenced by pickled artifacts can be imported during unpickling.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DIR = os.path.join(ROOT, 'saved_models')
for fn in sorted(os.listdir(DIR)):
    if fn.endswith('_artifacts.joblib'):
        path = os.path.join(DIR, fn)
        try:
            a = joblib.load(path)
            model = a.get('model')
            print(fn, '->', type(model), getattr(model, '__class__', None).__name__)
        except Exception as e:
            print(fn, '-> error loading:', e)
