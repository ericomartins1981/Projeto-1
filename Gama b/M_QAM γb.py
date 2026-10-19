import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Parâmetros da simulação
Nt = 100000                           # Quantidade total de símbolos por ponto de SNR
snr_db_range = np.arange(0, 26, 2)    # Valores de SNR por bit (Eb/N0) em dB
m_ordens = [4, 16, 64, 256]           # Ordens de modulação
cores = ['r', 'b', 'g', 'orange']     # Cores para cada curva

def gerar_constelacao(M):                # Função para gerar os pontos da constelação M-QAM normalizada para energia média = 1 
    sqrt_M = int(np.sqrt(M))
    m_levels = np.arange(-(sqrt_M - 1), sqrt_M + 1, 2)
    I, Q = np.meshgrid(m_levels, m_levels)
    constelacao = I.flatten() + 1j * Q.flatten()
    E_media = np.mean(np.abs(constelacao)**2)
    constelacao = constelacao / np.sqrt(E_media)
    return constelacao

plt.figure(figsize=(10, 7))

print("Iniciando simulações para as ordens M-QAM...")

for M, color in zip(m_ordens, cores):
    # 1. Gerar a constelação para o M atual
    constelacao = gerar_constelacao(M)
    sqrt_M = int(np.sqrt(M))
    
    ser_simulado = []
    ser_teorico = []
    
    print(f"Simulando {M}-QAM...")
    
    for snr_db in snr_db_range:
        # Converter Eb/N0 de dB para linear e relacionar com Es/N0 (Es = Eb * log2(M))
        eb_n0_linear = 10**(snr_db / 10.0)
        snr_linear = eb_n0_linear * np.log2(M)
        
        # Potência do ruído para energia média unitária
        No_power = 1.0 / snr_linear
        No_std = np.sqrt(No_power / 2.0)
        
        # 2. Gerar símbolos transmitidos aleatórios
        random_indices = np.random.randint(0, M, size=Nt)
        sinal_TX = constelacao[random_indices]
        
        # 3. Adicionar ruído AWGN complexo
        No = No_std * (np.random.randn(Nt) + 1j * np.random.randn(Nt))
        sinal_RX = sinal_TX + No
        
        # 4. Comparação entre o que foi transmitido com o que foi decodificado
        d = np.abs(sinal_RX[:, None] - constelacao[None, :])
        simbolos_detec = np.argmin(d, axis=1)
        
        # 5. Cálculo da Probabilidade de Erro de Símbolo (SER Simulado)
        Ne = np.sum(random_indices != simbolos_detec)
        ser = Ne / Nt
        ser_simulado.append(ser)
        
        # 6. Cálculo da SEP Teórica para M-QAM
        arg = np.sqrt(3.0 * snr_linear / (M - 1.0))
        q_val = 0.5 * erfc(arg / np.sqrt(2.0))
        p_1d = 2.0 * (1.0 - 1.0 / sqrt_M) * q_val
        ser_theo = 1.0 - (1.0 - p_1d)**2
        ser_teorico.append(ser_theo)

    # Plotagem da Curva Teórica (Linha contínua) 
    plt.semilogy(snr_db_range, ser_teorico, color=color, linestyle='-', linewidth=2, label=f'SEP Teórica ({M}-QAM)')
    
    # Plotagem da Curva Simulada (Linha tracejada em preto) 
    plt.semilogy(snr_db_range, ser_simulado, color='k', linestyle='--', linewidth=1.5, label=f'SER Simulação ({M}-QAM)')

# ==========================================
# CONFIGURAÇÃO DO GRÁFICO
# ==========================================
plt.title('SEP e SER M-QAM', fontsize=12)
plt.xlabel('SNR por Bit - $\gamma_b$', fontsize=11)
plt.ylabel('Probabilidade de Erro', fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=8, loc='lower left', ncol=2)
plt.ylim([1e-4, 1.1])
plt.show()