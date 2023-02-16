# Tacotron TTS implementation using pytorch

### Training

1. **환경 설정**

    * 해당 repository를 clone 합니다.
    * anaconda 환경 생성 후, 필요한 패키지를 설치 합니다.

    ```
    git clone https://github.com/kiple0219/Tacotron-pytorch.git
    conda create -n tacotron python=3.8.5
    conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt
    ```


2. **학습 데이터 준비**

     * dataset 디렉토리에 학습을 위한 wav음성 데이터와 텍스트 엑셀 파일을 준비합니다.
     * 텍스트 엑셀 파일은 1열에 wav 파일 이름, 2열에 해당 텍스트 문장을 저장합니다.
   ```
   Tacotron-pytorch
     |- dataset
         |- my_wav
         |- my_text.xlsx
   ```

3. **Preprocess**
   ```
   python preprocess.py
   ```
     * data 디렉토리에 학습에 필요한 파일들이 생성됩니다.

4. **Train**
   ```
   python train1.py -n <name>
   python train2.py -n <name>
   ```
     * train1.py - train2.py 순으로 실행합니다.
     * 원하는 name을 정하면 ckpt/<name> 디렉토리가 생성되고 모델이 저장됩니다.
     * 재학습은 아래와 같이 로드할 모델 경로를 정해주면 됩니다.
  
   ```
   python train1.py -n <name> -c ckpt/<name>/1/ckpt-<step>.pt
   python train2.py -n <name> -c ckpt/<name>/2/ckpt-<step>.pt
   ```
  
5. **Synthesize**
   ```
   python test1.py -c ckpt/<name>/1/ckpt-<step>.pt
   python test2.py -c ckpt/<name>/2/ckpt-<step>.pt
   ```
     * test1.py - test2.py 순으로 실행하면 output 디렉토리에 wav 파일이 생성됩니다.
 
### Reference
   https://github.com/chldkato/Tacotron-pytorch  
   https://github.com/soobinseo/Tacotron-pytorch
