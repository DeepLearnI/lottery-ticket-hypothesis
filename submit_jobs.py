import foundations as f9s

NUM_JOBS = 5

for i in range(NUM_JOBS):
    f9s.deploy(env='scheduler',
               entrypoint='new_main.py',
               project_name='Helen - Satellite image segmentation')