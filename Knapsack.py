import numpy as np

def Knapsack(weights, values, capacity):
    n = len(weights)
    dp = np.zeros((n + 1, capacity + 1), dtype=float)
    choose = np.zeros((n + 1, capacity + 1), dtype=bool)

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                # dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                if dp[i - 1][w] <= dp[i - 1][w - weights[i - 1]] + values[i - 1]:
                    dp[i][w] = dp[i - 1][w - weights[i - 1]] + values[i - 1]
                    choose[i][w] = True
            else:
                dp[i][w] = dp[i - 1][w]
                
    np_items = []
    w = capacity
    for i in range(n, 0, -1):
        if choose[i][w]:
            np_items.append(i - 1)  # Store the index of the item
            w -= weights[i - 1]  # Reduce the weight by the weight of the chosen item
            
    #print sum weights 
    # print(f"Total weight of selected items: {sum(weights[i] for i in np_items)} / {capacity}")
            
    return dp[n][capacity], np_items[::-1]  # Return the maximum value and the items in reverse order