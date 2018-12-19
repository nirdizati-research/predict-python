import time

from encoders.boolean_frequency import frequency
from logs.file_service import get_logs

start_time = time.time()
log = get_logs('/Users/tonis.kasekamp/other/predict-python/logdata/Production.xes')[0]
print("xes to object done in %s seconds ---" % (time.time() - start_time))

df = frequency(log)  # TODO: fix arguments passing or remove file
print("Total %s seconds" % (time.time() - start_time))
print(df.shape)
print(df.columns.values)
