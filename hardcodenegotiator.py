prompt=10000
original=prompt
userPrice=0
while(True):
    print('The item price is: ',prompt)
    userPrice = int(input("Are you happy with this price(enter -1 if so, provide your expected price otherwise): "))
    if(userPrice == -1):
        print("Great! The final price is: ", prompt)
        break
    if(userPrice<0.5*original):
        print("I appreciate your offer, however I cannot accept it. Do you have a better offer?")
        userPrice = int(input("Please enter your new offer: "))
    if(userPrice>=0.5*original and userPrice<0.75*original):
        prompt=0.95*original

    elif(userPrice>=0.75*original and userPrice<0.9*original):
        prompt=0.9*original
    elif(userPrice>=0.9*original and userPrice<0.95*original):
        prompt=userPrice
        print("It's a deal! The final price is: ", prompt)
        break