import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F






def main():


    rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=1)
    print(rnn)
    x = torch.randn(10, 3, 100)
    out, h = rnn(x, torch.zeros(1, 3, 20))
    print(out.shape, h.shape)

    rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=4)
    print(rnn)
    x = torch.randn(10, 3, 100)
    out, h = rnn(x, torch.zeros(4, 3, 20))
    print(out.shape, h.shape)
    # print(vars(rnn))

    print('rnn by cell')

    cell1 = nn.RNNCell(100, 20)
    h1 = torch.zeros(3, 20)
    for xt in x:
        h1 = cell1(xt, h1)
    print(h1.shape)


    cell1 = nn.RNNCell(100, 30)
    cell2 = nn.RNNCell(30, 20)
    h1 = torch.zeros(3, 30)
    h2 = torch.zeros(3, 20)
    for xt in x:
        h1 = cell1(xt, h1)
        h2 = cell2(h1, h2)
    print(h2.shape)

    print('Lstm')
    lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=4)
    print(lstm)
    x = torch.randn(10, 3, 100)
    out, (h, c) = lstm(x)
    print(out.shape, h.shape, c.shape)

    print('one layer lstm')
    cell = nn.LSTMCell(input_size=100, hidden_size=20)
    h = torch.zeros(3, 20)
    c = torch.zeros(3, 20)
    for xt in x:
        h, c = cell(xt, [h, c])
    print(h.shape, c.shape)


    print('two layer lstm')
    cell1 = nn.LSTMCell(input_size=100, hidden_size=30)
    cell2 = nn.LSTMCell(input_size=30, hidden_size=20)
    h1 = torch.zeros(3, 30)
    c1 = torch.zeros(3, 30)
    h2 = torch.zeros(3, 20)
    c2 = torch.zeros(3, 20)
    for xt in x:
        h1, c1 = cell1(xt, [h1, c1])
        h2, c2 = cell2(h1, [h2, c2])
    print(h2.shape, c2.shape)






if __name__ == '__main__':
    main()