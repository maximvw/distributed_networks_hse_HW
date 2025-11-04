import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('results/task1_pi_results.csv')
print(df)
t1 = df[df["procs"]==1].time.values[0]
df['speedup'] = t1 / df.time
df['eff'] = df['speedup'] / df.procs
plt.figure()
plt.plot(df.procs, df.time, '-o')
plt.xlabel('procs'); plt.ylabel('time (s)')
plt.title('time vs procs')
plt.savefig('graphics/time_vs_procs.png')

plt.figure()
plt.plot(df.procs, df.speedup, '-o')
plt.xlabel('procs'); plt.ylabel('speedup')
plt.title('speedup = t(1) / t(p) vs procs')
plt.savefig('graphics/speedup_vs_procs.png')

plt.figure()
plt.plot(df.procs, df.eff, '-o')
plt.xlabel('procs'); plt.ylabel('efficiency')
plt.title('efficiency = speedup / procs vs procs')
plt.savefig('graphics/efficiency_vs_procs.png')

plt.show()
