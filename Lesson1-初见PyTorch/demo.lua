torch.Tensor{1,2,3}

function gradUpdate(mlp,x,y,learningRate)
  local criterion = nn.ClassNLLCriterion()
  pred = mlp:forward(x)
  local err = criterion:forward(pred, y); 
  mlp:zeroGradParameters();
  local t = criterion:backward(pred, y);
  mlp:backward(x, t);
  mlp:updateParameters(learningRate);
end
