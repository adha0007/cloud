## Object detection service

### Prerequisite
* Pycharm or similar IDE
* Virtualenv setup
* Python3.7

##Getting Started
### To setup virtualenv
1) Create separate folder outside repo folder e.g "env" folder.
2) Inside that env folder run following
    > pip3 install virtualenv
           
    > virtualenv -p python3 detect_env
3) To activate this virtualenv hit one of the command.
    > source detect_env/bin/activate 
                                                                                                            
    Now from your project root dir, run following                                                                                                            

    > pip3 install -r requirements.txt
                                                                                                            
To stop this virtualenv, just type (whenever dont want to use)
    
> deactivate                        
                    
### Other requirements
    1. Redis server on local 
    2. DynamoDB on local https://www.dynamodbguide.com/
    3. Create .env file and copy data from env.default into it
                                      
### To run this project
 Run detect.py file to start the api

