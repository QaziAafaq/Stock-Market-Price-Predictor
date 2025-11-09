def sliding_window_prediction(prices, no_pre):
    window_size = 5
    predictions = []

    for i in range(no_pre):
        window = prices[-window_size:]            
        prediction = sum(window) / len(window)    
        predictions.append(prediction)            
        prices.append(prediction)             

    return predictions

prices = [110, 103, 109, 110, 145]
while(True):
    n = input("Enter number of future predictions: ")
    try:
        no_pre = int(n)
        predicted_prices = sliding_window_prediction(prices, no_pre)
        print(predicted_prices)
    except:
        if n == 'exit':
            break
        else:
            print("Error")
