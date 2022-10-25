# 2022 APAC HPC-AI Competition
## Team NTHU-1

> Gadi path of this repo: `/scratch/jx00/2022-APAC-HPC-AI`
> 
> [GitHub link](https://github.com/nevikw39/2022-APAC-HPC-AI/)

Report links:
1. [HPC with QE](2022_APAC_HPC-AI_Team_NTHU-1_High_Performance_Computing_With_Quantum_Espresso.pdf)
2. [Communications Performance with UCX](2022_APAC_HPC-AI_Team_NTHU-1_Communication_Performance_with_UCX.pdf)
3. [DL-based DNA Sequence Fast Decoding](2022_APAC_HPC-AI_Team_NTHU-1_Deep-Learning-based_DNA_Sequence_Fast_Decoding.pdf)

And here is presentation [link](2022_APAC_HPC-AI_Team_NTHU-1_Presentation.pdf).

## HPC with QE

### Job Scripts

- **Single node**: [High_Performance Computing_with_Quantum_Espresso/CeO2_qe_input/qe-single.sh](High_Performance%20Computing_with_Quantum_Espresso/CeO2_qe_input/qe-single.sh)
- **Multiple nodes**: [High_Performance Computing_with_Quantum_Espresso/CeO2_qe_input/qe-multi.sh](High_Performance%20Computing_with_Quantum_Espresso/CeO2_qe_input/qe-multi.sh)

### Result Logs

- **Single node**: [High_Performance Computing_with_Quantum_Espresso/qe-single.log](High_Performance%20Computing_with_Quantum_Espresso/qe-single.log)
- **Multiple nodes**: [High_Performance Computing_with_Quantum_Espresso/qe-multi.log](High_Performance%20Computing_with_Quantum_Espresso/qe-multi.log)

## Communications Performance with UCX

Environment installation is the same as original task.

Our optimized configurations has beed appended to [Communications_Performance_with_UCX/cluster.cfg](Communications_Performance_with_UCX/cluster.cfg). They could also be `source`d from [Communications_Performance_with_UCX/optimized.cfg](Communications_Performance_with_UCX/optimized.cfg).

### Result Outputs

- **Small data set**: [Communications_Performance_with_UCX/small_data_set.out](Communications_Performance_with_UCX/small_data_set.out)
- **Large data set**: [Communications_Performance_with_UCX/small_data_set.out](Communications_Performance_with_UCX/large_data_set.out)

## DL-Based DNA Sequence Fast Decoding

Environment installation is the same as original task.

The Leopard Unet model source code has been appended to [Deep_Learning_Based_DNA_Sequence_Fast_Decoding/models.py](Deep_Learning_Based_DNA_Sequence_Fast_Decoding/models.py). Its output is in [Deep_Learning_Based_DNA_Sequence_Fast_Decoding/output](Deep_Learning_Based_DNA_Sequence_Fast_Decoding/output). The output log is [Deep_Learning_Based_DNA_Sequence_Fast_Decoding/Leopard_NCI.log](Deep_Learning_Based_DNA_Sequence_Fast_Decoding/Leopard_NCI.log).

### Hyperparameters

- **Batch size**: 64
- **Learning rate**: 0.01
- **# GPU**: 1
