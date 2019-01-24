import time

from encoders.boolean_frequency import frequency
from logs.file_service import get_logs_old

start_time = time.time()
log = get_logs_old('/Users/tonis.kasekamp/other/predict-python/logdata/Production.xes')[0]
print("xes to object done in %s seconds ---" % (time.time() - start_time))

df = frequency(log)
print("Total %s seconds" % (time.time() - start_time))
print(df.shape)
print(df.columns.values)
