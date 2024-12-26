models used in the mlops examples are basic cnn and pre trained resnet 18 models
backend by flask
frontend html,css,javascript
and finally the ai models and training done by pytorch

the model path for basic cnn is present in /model but for resnet as it is larger than 25 mb i have it in https://drive.google.com/file/d/1-7jj99XSlraXYQb9DHKfOVnPUSJPzgev/view?usp=drive_link

the process involved :
fetching the dataset
  using the cifar-10 dataset and isolating cats 3 and dogs 5 from them,
preprocessing data 
  for use by the models (normalization for pre trained ones obtained from the web), 
constructing 
  the models weights and forward function,
training the models
  passing the models through 80 cycles with BCE (for binary classification) and SGD,

Resources used :
https://flask.palletsprojects.com/en/stable/patterns/fileuploads/
https://www.youtube.com/watch?v=9L9jEOwRrCg&t=320s
https://www.learnpytorch.io/03_pytorch_computer_vision/
https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/capture
https://www.geeksforgeeks.org/how-to-avoid-overfitting-in-machine-learning/
https://www.youtube.com/watch?v=nVhau51w6dM&t=4168s
https://www.youtube.com/watch?v=bluclMxiUkA

