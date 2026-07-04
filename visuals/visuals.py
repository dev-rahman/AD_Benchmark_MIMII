import matplotlib.pyplot as plt
import librosa.display

fig, ax = plt.subplots(2, 1, figsize=(6, 2.8), sharex=True)

librosa.display.waveshow(y_normal, sr=sr, ax=ax[0])
ax[0].set_title("Normal", fontsize=9)
ax[0].set_ylabel("Amp.", fontsize=8)
ax[0].tick_params(axis='both', labelsize=7)

librosa.display.waveshow(y_anomaly, sr=sr, ax=ax[1])
ax[1].set_title("Anomaly", fontsize=9)
ax[1].set_ylabel("Amp.", fontsize=8)
ax[1].set_xlabel("Time (s)", fontsize=8)
ax[1].tick_params(axis='both', labelsize=7)

plt.tight_layout(pad=0.6)
plt.savefig("fan_waveforms_clean.png", dpi=300, bbox_inches="tight")
plt.show()