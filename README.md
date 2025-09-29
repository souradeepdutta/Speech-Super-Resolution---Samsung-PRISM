# Audio Super-Resolution using ECA-Enhanced Encoder-Decoder CNN

This project implements a deep learning model for audio super-resolution, specifically designed to enhance low-sampling-rate speech signals for improved performance in Automatic Speech Recognition (ASR) systems. The model upsamples audio from 2, 4, or 8 kHz to 16 kHz.



## Dataset

The model was trained and evaluated on the **LibriSpeech** dataset, using 100 hours of the `train-clean-100` subset.



## Methodology

The approach is centered around a lightweight, efficient neural network that reconstructs high-frequency details lost in low-resolution audio.

### 1. Data Preprocessing

1.  **Downsampling**: The original 16 kHz audio from LibriSpeech is downsampled to 2 kHz, 4 kHz, and 8 kHz to serve as the low-resolution inputs.
2.  **Feature Extraction**: Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from both the low-resolution and the original high-resolution audio files.
3.  **Normalization**: The extracted features are standardized (mean=0, std=1) before being fed into the model.

### 2. Model Architecture

The core of the project is a U-Net-like encoder-decoder architecture built with 1D Convolutional layers.

* **Encoder**: A series of `Conv1D` layers with `ReLU` activation and `BatchNorm` capture temporal features and downsample the input.
* **Efficient Channel Attention (ECA)**: After key convolutional blocks in the encoder, ECA modules are used. These lightweight attention modules allow the network to focus on the most informative feature channels without adding significant computational overhead.
* **Decoder**: `Conv1DTranspose` layers are used to upsample the features and reconstruct the high-resolution signal. Skip connections are present between corresponding encoder and decoder layers to preserve finer details.



### 3. Training

The model is trained to optimize a combined loss function that balances signal fidelity and perceptual quality for ASR:

* **Signal Reconstruction Loss**: Mean Squared Error (MSE) between the predicted and the ground-truth high-resolution audio.
* **ASR-aware Perceptual Loss**: An L1 loss between the latent representations of the predicted and ground-truth audio, extracted from a pre-trained **Wav2Vec 2.0** model. This pushes the model to generate output that is more intelligible to ASR systems.



## Results

The ECA-enhanced CNN was benchmarked against traditional interpolation methods and other attention-based models. Performance was measured using ASR metrics (Word Error Rate - WER, Character Error Rate - CER) and audio quality metrics (PESQ, STOI).

| Method | WER (%) | CER (%) | PESQ | STOI |
| :--- | :--- | :--- | :--- | :--- |
| **ECA-CNN (Ours)** | **26.67** | **13.92** | **1.62** | **0.80** |
| SE-CNN | 29.71 | 15.05 | 1.68 | 0.80 |
| CBAM-CNN | 42.79 | 24.20 | 1.06 | 0.56 |
| Cubic Interpolation| 28.37 | 14.51 | 1.70 | 0.80 |
| Linear Interpolation| 32.69 | 16.69 | 1.45 | 0.80 |
| Nearest Interpolation| 33.58 | 17.59 | 1.08 | 0.79 |

_Table data sourced from the results table in the provided context._

### Conclusion

The **ECA-CNN** model provides the best balance of performance and efficiency. While traditional methods like Cubic Interpolation achieve high perceptual scores (PESQ), they result in poor ASR accuracy (higher WER/CER). The ECA-CNN significantly improves ASR performance over baseline methods while remaining computationally efficient, making it a practical solution for enhancing speech quality in ASR pipelines.



## How to Run

The entire workflow, from preprocessing to training, is contained within the `super-resolution-speech-final.ipynb` Jupyter Notebook.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/souradeepdutta/Speech-Super-Resolution-Samsung-PRISM.git](https://github.com/souradeepdutta/Speech-Super-Resolution-Samsung-PRISM.git)
    cd your-repo-name
    ```

2.  **Set up the Environment:**
    * Install the necessary libraries listed in the first cell of the notebook (e.g., `tensorflow`, `librosa`, `soundfile`).
    * Download the LibriSpeech dataset and update the file paths in the notebook cells to point to your dataset's location.

3.  **Execute the Notebook:**
    * Open `super-resolution-speech-final.ipynb` in a Jupyter environment (like Jupyter Lab, VS Code, or Google Colab).
    * Run the cells sequentially from top to bottom to perform:
        * **Data Preprocessing**: Converting audio to low frequency, extracting features, and normalization.
        * **Dataset Creation**: Building the `tf.data.Dataset` for training.
        * **Model Training**: Compiling and training the super-resolution model.
