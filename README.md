# Real-time-Emotion-Detection-and-Speech-Response-Generation
# Our basic orientation:
REDSG is a lightweight system that can recognize emotion from camera, and offers flexible voice responses according to emotion changes simultaneously.

# Accurate emotion recognition effect：
<img width="480" alt="35e9cd93cc59b30dab1c8af7dfec5c7" src="https://github.com/user-attachments/assets/46cd2ca4-5dc7-4bbe-b1c0-69cc2d2cf568" /><img width="480" alt="f41bb7c323a675349e5b1d5e82f0175" src="https://github.com/user-attachments/assets/ad67809f-edf5-4390-8424-b187b85c85aa" />
<img width="480" alt="ef36742ea2d1625ae7d534d1de73c86" src="https://github.com/user-attachments/assets/43ce4ebc-4d01-41c2-abc8-9b96099111da" /><img width="480" alt="41dab9ddb38c79e5996fc144111defd" src="https://github.com/user-attachments/assets/e06a72fd-0fe8-4ba8-b35c-9dd7703ee557" /><img width="480" alt="493b0f0545e2f21983776f64ac9245e" src="https://github.com/user-attachments/assets/ec9db0a4-857c-44c0-8a57-2098292e9422" />
<img width="480" alt="4905ce2bc7bda105001aad9347f290e" src="https://github.com/user-attachments/assets/58f8f526-2130-4c3b-80c4-126328caffd9" />


# Diverse reply outputs：
![image](https://github.com/user-attachments/assets/bb7800da-734d-4dc7-a4e9-bf18819012d0)
![image](https://github.com/user-attachments/assets/655598b0-e8b6-4e47-8b59-deacd902d4fb)
![image](https://github.com/user-attachments/assets/39ff42bf-32e2-401f-b1bb-ef27caf0c802)
![image](https://github.com/user-attachments/assets/2541228b-0f78-4bbb-a690-8b8d4005258f)
![image](https://github.com/user-attachments/assets/cb05b5fa-8969-450f-8c18-db84d7028f4e)
![image](https://github.com/user-attachments/assets/9d8bac88-134d-4aa8-8469-d3ee5f06fc7c)


# Rapid Testing:
## Step1:
<img width="272" alt="bc13208f6af280675127b8ca6a03536" src="https://github.com/user-attachments/assets/13d9028e-ea5a-476f-a37f-b56464752327" />
Click "code" to download the zip file.
## Step2:
Unzip the zipped file and open the folder with VS Code.
## Step3:
Configure your OpenAI key.
## Step4:
Configure the virtual environment according to the requirements.txt in the requirements file, enter the file path in the terminal, and run the Python EM_p.py.


# Ideas & Architecture：





# Our advantages:
## 1.We use the cutting-edge DeepFace library for sentiment analysis.
### Why we choose Deepface for my project?
a. **Efficient facial emotion recognition**: DeepFace provides efficient facial emotion recognition capabilities, able to process video streams in real-time and accurately identify emotional states, meeting the facial emotion recognition accuracy requirement (≥75%).
b. **Multimodal support**: DeepFace can easily integrate with other emotion analysis models, effectively complementing voice emotion recognition and improving the overall system performance (voice ≥80%).
c. **Fast response and low latency**: DeepFace optimizes the facial recognition algorithm, supporting rapid emotion recognition processing and meeting real-time requirements (end-to-end latency ≤300ms).
d. **Lightweight**: DeepFace is lightweight and optimized, making it suitable for deployment in resource-constrained environments without compromising performance.

## 2.We use OpenAI's GPT-3.5-Turbo large language model to track sentiment and generate responses in real-time.
### Why we choose OpenAI's GPT-3.5-Turbo large language model for my project?
a. **Advanced language understanding**: GPT-3.5-Turbo can understand and generate complex conversational content, generating accurate dialogue strategies and responses based on real-time emotion analysis.
b. **Efficient response generation**: GPT-3.5-Turbo can quickly generate high-quality speech responses, meeting low latency requirements (end-to-end latency ≤ 300ms).
c. **Scalability and flexibility**: GPT-3.5-Turbo has strong scalability, capable of adjusting the response strategies of the speech interaction system according to different emotional states, adapting to various interaction scenarios.

## 3.We use pyttsx3 to convert text into real-time speech.
### Why we choose OpenAI's GPT-3.5-Turbo large language model for my project?
1. **Lightweight and efficient**: pyttsx3 is a lightweight text-to-speech library that can quickly generate speech output, meeting low latency requirements (end-to-end latency ≤ 300ms).
2. **Cross-platform support**: pyttsx3 supports cross-platform operation (Windows, macOS, Linux), making it easy to deploy the speech interaction system on various devices.
3. **Speech customization capabilities**: pyttsx3 allows customization of voice tone, speech rate, and volume to match different emotional states, enhancing the immersion of speech interactions.






    
