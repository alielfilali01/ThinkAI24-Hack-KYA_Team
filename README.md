### Problem Description
We aim to address the challenge of providing tourists and visitors with seamless access to relevant information and assistance during major events in Morocco, such as the Africa Cup of Nations and the World Cup 2030. Navigating a foreign country can be challenging, and current solutions often require sifting through multiple sources of information. Our solution is a chatbot assistant powered by a custom Vision Language Model (VLM) integrated with a Retrieval-Augmented Generation (RAG) system, designed to offer real-time, accurate, and context-aware assistance to users.

### Use Cases and Demo
1. **Navigational Assistance**: Show how the chatbot assists a tourist in finding the quickest route to a stadium using public transport options.
2. **Cultural Information**: Demonstrate the chatbot providing detailed information about a Moroccan dish by analyzing a photo taken by the user.
3. **Booking Services**: Display the chatbot's capability to book an Uber or other services directly through API calls.
4. **Multilingual Support**: Highlight the chatbot's ability to interact with users in their native languages using translation models.

### Interface (UI)
Our project includes a user-friendly interface that demonstrates the chatbot's capabilities. The interface allows users to interact with the chatbot in a seamless and intuitive manner. The structure of the interface is as follows:

1. **example_images**: A folder containing example images for the demo.
2. **YALLA_logo.png**: The logo of our YALLA model.
3. **app_dialogue.py**: The Python script to launch the Gradio-based demo interface.
4. **requirements.txt**: A file listing the necessary dependencies to run the interface.

To launch the interface, follow these steps:
1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the Gradio demo:
   ```
   python ./gradio_demo/app_dialogue.py
   ```
<!--
### Storyline of our 4-Minute Pitch + 2-Minute Q&A

1. **Slide 1: Introduction**
   - Brief overview of the problem and our solution.
   - Importance of enhancing tourist experience during major events.

2. **Slide 2: Use Cases and Benefits**
   - Navigational assistance, cultural information, booking services, and multilingual support.
   - How these features improve the overall experience for visitors.

3. **Slide 3: Technical Details**
   - Explanation of the Vision Language Model and RAG system.
   - Description of the dataset and finetuning process.

4. **Slide 4: Future Enhancements and Impact**
   - Potential integrations (e.g., cooperative and artisan data, tool usage).
   - Expected economic and cultural benefits for Morocco.  
-->

### Project Description for GitHub Repository
**GenAI Challenge Path:** Cultural Heritage

**Project Description:**
Our project aims to enhance the experience of tourists and visitors during major events in Morocco by providing a chatbot assistant powered by a custom Vision Language Model (VLM) integrated with a Retrieval-Augmented Generation (RAG) system. This chatbot offers real-time, context-aware assistance for navigation, cultural information, booking services, and more, all through a user-friendly interface. We believe this solution will not only improve visitor satisfaction but also promote Moroccan culture and contribute to the local economy by encouraging interactions with local businesses.

**Source Code / Notebooks:** Included in the repository.

**Trained Models / APIs:**
- Base Model: [HuggingFaceM4/idefics2-8b-chatty](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty)
- Custom Dataset: [MA-Culture-Vision-v0.2](https://huggingface.co/datasets/Ali-C137/MA-Culture-Vision-v0.2)
- Finetuned Model: [Ali-C137/idefics2-8b-yalla-finetuned-cultural](https://huggingface.co/Ali-C137/idefics2-8b-yalla-finetuned-cultural)

### **Demo:** 
 - Recorded video demonstrating the working example of our solution : https://youtu.be/TudD2-gOR8o

**Evaluation Protocol and Benchmarks:**
- Comparison of responses from 4 different VLMs against our finetuned version.
  - The prompt : **What can you tell me about this picture**
  - The image :
 
    ![image](https://github.com/alielfilali01/ThinkAI24-Hack-KYA_Team/assets/58257628/7717672a-88d8-4999-b985-6de608ff5978)


  - The Responses : 
    - DeepSeek-VL-7B :
      
      - The image showcases a pair of intricately designed wooden doors set within an ornate archway. The doors are adorned with detailed geometric patterns and Arabic calligraphy, suggesting cultural or historical significance. The archway is embellished with blue tiles featuring a repeating diamond pattern, which contrasts beautifully with the warm tones of the wood. The overall aesthetic appears to be influenced by Islamic art and architecture, characterized by symmetry, repetition, and intricate craftsmanship. The lighting in the photo highlights the textures and details of both the doors and the surrounding tilework, creating a sense of depth and dimension.
 
    
    - Qwen-VL-Max (API) :
       - The image presents an architectural marvel, a door that is a testament to intricate craftsmanship and design. The door, made of wood, stands out with its gold color, adorned with a pattern of interlocking circles and squares. It's not just the door but also the surrounding wall that adds to the visual appeal. The wall, painted in blue, features a pattern of white lines and dots, creating a striking contrast against the gold door.
The door is framed by an archway, which further enhances its grandeur. Two handles, each featuring a unique design, are attached to the door, adding to its aesthetic appeal. The image does not contain any text or other discernible objects. The relative position of the objects suggests that the door is the central focus of the image, with the wall serving as a backdrop that complements it beautifully. The image exudes a sense of symmetry and balance, with every element meticulously placed to create a harmonious composition.

    -  PaliGemma :
        - In this image we can see a door.
    
    -   IDEFICS2 :
           - This is a door. In the background there are walls with designs on it.
    
    - **YALLA (ours)** :
      - This image captures a beautiful Moroccan door set against a backdrop of blue and gold tiles. The door itself is made of wood and features a gold-colored metal design. It has two small handles on either side, adding to its charm. The door is framed by a large archway, which is adorned with a gold-colored metal lattice design. The overall effect is one of opulence and cultural richness. 
             ![image](https://github.com/alielfilali01/ThinkAI24-Hack-KYA_Team/assets/58257628/731cead2-43b1-44a2-b5ea-e6829ef991a2)

### Limitation and Future Perspectives

#### Limitations
1. **Data Coverage and Accuracy**: Although our chatbot is powered by a robust Vision Language Model and extensive datasets, there might be gaps in data coverage or inaccuracies due to the dynamic nature of cultural and logistical information.
2. **Language Support**: While our solution should be able to support multiple languages through translation models (NLLB), there could be nuances and dialects that are not fully captured, potentially leading to misunderstandings.

#### Future Perspectives
1. **Enhanced Data Integration**: Expand the dataset to include more comprehensive and up-to-date information on Moroccan culture, logistics, and events. This includes integrating real-time data sources for transport, events, and local services.
2. **Advanced Language Models**: Continuously update and fine-tune the language models to support more languages and dialects, ensuring more accurate and context-aware responses.
3. **Improved User Experience**: Develop a more interactive and intuitive user interface, incorporating feedback from users to enhance usability and satisfaction.
4. **Extended Functionalities**: Introduce new features such as personalized recommendations, local business promotions, and interactive maps to provide a richer and more engaging user experience.
5. **Partnerships and Collaborations**: Collaborate with local businesses, government agencies, and event organizers to ensure the chatbot provides the most relevant and useful information, promoting local culture and economy effectively.

By addressing these limitations and exploring future perspectives, our solution aims to become an indispensable tool for enhancing the experience of tourists and visitors, showcasing the rich culture of Morocco, and supporting the local economy during major events.
