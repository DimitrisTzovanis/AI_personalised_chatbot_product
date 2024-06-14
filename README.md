# AI personalised chatbot full product

## Members: Dimitris Tzovanis, Dimitris Karydis

#### This project includes a front-end android application for the user interface, and 2 back-end servers. One java server for direct user communication, and one python server for AI model creation and execution.

#### This project contains the following folders:
- idea ai application: full andorid studio project of the user application. Users can manage their account and create their personal models by sending through the application their train data. Users can also chat with their AI models by typing their prompts.
- java_server: multithreaded java server project for direct user communication. Supports functions such as login / register / pull data. Also communicates with the python server for any actions regarding the AI.
- models: this folder contains all the created AI models (we havent included any as the model files are too large), and a python script which is a multithreaded python server that connects only with the previously mentioned java server. It supports model creation on data that the user sent (finetuning), and model exectuion on user prompts
  
#### Login info: user1 / pass1
