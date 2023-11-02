I1=[-10,-8,-6,-4,-2,0.00001,2,4,6,8,10];
I=zeros(2,121);
for i=1:11
    for j=1:11
        I(:,11*(i-1)+j)=[I1(i);I1(j)];
    end
end
q=f4(I(1,:),I(2,:))
[t,p]=size(I);
n=20;%隐含层的神经元个数
m=3;%层数
w=cell(1,m); %置各权值的初始值为小的随机数
w{1}=0.1*rand(n,2);
w{m}=0.1*rand(1,n);
for i=2:m-1
    w{i}=0.1*rand(n,n);
end
x=cell(1,m);%每层输出值
d=cell(1,m);%每层输出误差值
g=cell(1,m);%每层阈值
g{m}=0.1*rand(1,1); %置各神经元阈值的初始值为小的随机数
for i=1:m-1
    g{i}=0.1*rand(n,1);
end
a=0.2; %学习率
for j=1:40000 %设置循环次数
e=0; 
for k=1:121  %11*11个训练样本，依次对其学习
    x=forward(w,I,k,g,m);%计算网络各层的实际输出
    d=backward(k,m,x,q,w);%计算训练误差
    w{1}= w{1}+a*d{1}*I(:,k)';%修正权值
    for i=2:m
        w{i}=w{i}+a*d{i}*x{i-1}';
    end
    for i=1:m         %修正阈值
        g{i}=g{i}+a*d{i};
    end
end
for k=1:121 %计算性能指标
    x=forward(w,I,k,g,m);  
    ep=0.5*(q(k)-x{m})^2;
    e=e+ep;
end
e
E(j)=e;
N(j)=j;
if(e<0.001) %设置性能指标要求
    break;
end
end
I1=-10:1.0:10; %画检验样本曲面
I=zeros(2,21*21);
for i=1:21
    for j=1:21
        I(:,21*(i-1)+j)=[I1(i);I1(j)];
    end
end

for k=1:21*21
    x=forward(w,I,k,g,m);
    z(k)=x{m};
end
figure(1)
[x1,x2]=meshgrid(-10:1.0:10,-10:1.0:10);
z1=reshape(z,21,21);
mesh(x1,x2,z1);
%加入XY坐标轴和legend
xlabel('x1')
ylabel('x2')
zlabel('z')

figure(2)
%绘制和目标函数差值的曲面
z2=reshape(z,21,21);
z3=f4(x1,x2);
z4=z2-z3;
mesh(x1,x2,z4);
xlabel('x1')
ylabel('x2')
zlabel('z-z3')


figure(3)
plot(E)
%加入XY坐标轴和legend
xlabel('Epoch')
ylabel('MSE')
legend('MSE-log')

%函数4：
function y=f4(x1,x2)
y=sin(x1)./x1.*sin(x2)./x2;
end
%计算网络各层的输出：
function xt=forward(w,I,k,g,m)
xt=cell(1,m);
xt{1}=logsig(w{1}*I(:,k)+g{1});
for i=2:m-1
    xt{i}=logsig(w{i}*xt{i-1}+g{i});
end
xt{m}=w{m}*xt{m-1}+g{m};
end

%计算网络各层的输出误差：
function d=backward(k,m,x,q,w)
d=cell(1,m);
d{m}=q(k)-x{m};
for i=m-1:-1:1
    d{i}=x{i}.*(1-x{i}).*(w{i+1}'*d{i+1});
end
end
