import pandas as pd
import matplotlib.pyplot as plt
files = ['task1_pi_results', 
         'task2_matvec_block_results', 
         'task2_matvec_cols_results', 
         'task2_matvec_rows_results', 
         'task3_cannon_results', 
         'task4_dirichlet_results']
for file_name in files:
    df = pd.read_csv(f'results/{file_name}.csv')
    print(df)
    t1 = df[df["procs"]==1].time.values[0]
    df['speedup'] = t1 / df.time
    df['eff'] = df['speedup'] / df.procs
    plt.figure()
    plt.plot(df.procs, df.time, '-o')
    plt.xlabel('procs'); plt.ylabel('time (s)')
    plt.title('time vs procs')
    plt.savefig(f'graphics/{file_name}_time_vs_procs.png')

    plt.figure()
    plt.plot(df.procs, df.speedup, '-o')
    plt.xlabel('procs'); plt.ylabel('speedup')
    plt.title('speedup = t(1) / t(p) vs procs')
    plt.savefig(f'graphics/{file_name}_speedup_vs_procs.png')

    plt.figure()
    plt.plot(df.procs, df.eff, '-o')
    plt.xlabel('procs'); plt.ylabel('efficiency')
    plt.title('efficiency = speedup / procs vs procs')
    plt.savefig(f'graphics/{file_name}_efficiency_vs_procs.png')

    #plt.show()
