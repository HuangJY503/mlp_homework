% 定义输入数据
X = [0.25, 0.25; 0.75, 0.75; 0.75, 0.125; 0.25, 0.5; 0.5, 0.5; 0.75, 0.5; 0.25, 0.75; 0.5, 0.125; 0.75, 0.25]';
y = [2, 2, 2, 1, 1, 1, 0, 0, 0]';  % 0 for stars, 1 for moons, 2 for suns

% 初始化参数
input_size = size(X, 1);
hidden1_size = 12;
hidden2_size = 12;
output_size = 3;

W1 = randn(hidden1_size, input_size);
b1 = zeros(hidden1_size, 1);
W2 = randn(hidden2_size, hidden1_size);
b2 = zeros(hidden2_size, 1);
W3 = randn(output_size, hidden2_size);
b3 = zeros(output_size, 1);

% 定义激活函数
sigmoid = @(x) 1./(1+exp(-x));


% 定义学习率和训练轮数
learning_rate = 0.05;
num_epochs = 20000;

% 训练模型
for epoch = 1:num_epochs
    % 前向传播
    z1 = W1*X + b1;
    a1 = tanh(z1);
    z2 = W2*a1 + b2;
    a2 = tanh(z2);
    z3 = W3*a2 + b3;
    a3 = tanh(z3);
    
    % 计算损失
    one_hot_y = zeros(output_size, length(y));
    for i = 1:length(y)
        one_hot_y(y(i)+1, i) = 1;
    end
    loss = -sum(sum(one_hot_y.*log(a3) + (1-one_hot_y).*log(1-a3))) / length(y);
    
    % 反向传播
    delta3 = a3 - one_hot_y;
    delta2 = (W3'*delta3) .* a2 .* (1-a2);
    delta1 = (W2'*delta2) .* a1 .* (1-a1);
    
    % 计算梯度
    dW3 = (1/length(y)) * delta3 * a2';
    db3 = (1/length(y)) * sum(delta3, 2);
    dW2 = (1/length(y)) * delta2 * a1';
    db2 = (1/length(y)) * sum(delta2, 2);
    dW1 = (1/length(y)) * delta1 * X';
    db1 = (1/length(y)) * sum(delta1, 2);
    
    % 更新参数
    W3 = W3 - learning_rate * dW3;
    b3 = b3 - learning_rate * db3;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    
    % 打印损失
    if mod(epoch, 100) == 0
        disp(['Epoch ', num2str(epoch), ', Loss: ', num2str(loss)]);
    end


end

% 前向传播
z1 = W1*X + b1;
a1 = tanh(z1);
z2 = W2*a1 + b2;
a2 = tanh(z2);
z3 = W3*a2 + b3;
a3 = tanh(z3);

% 前向传播
z1 = W1*X + b1;
a1 = tanh(z1);
z2 = W2*a1 + b2;
a2 = tanh(z2);
z3 = W3*a2 + b3;
a3 = tanh(z3);

% 输出结果
[~, predicted_labels] = max(a3);
disp('Predicted Labels:');
disp(predicted_labels - 1);  % Convert back to 0 for stars, 1 for moons, 2 for suns

% 可视化
figure(1);
gscatter(X(1,:), X(2,:), predicted_labels, 'rgb', 'osd');
xlabel('X');
ylabel('Y');
title('Predicted Labels');
legend('Stars', 'Moons', 'Suns');