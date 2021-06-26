function c=middle(n,a,b,lambda)%n为核函数的近似阶数,m为区间剖分数
%a=0;b=1;lambda=5;n=5;
syms u t
y=const_y(lambda,a,b);
y=subs(y,t,u);
I1=[];
for j=1:n
    i=1:n;i=i';
    f=t^(j-1)/factorial(j-1)*t.^(i-1);%alafa与beta乘积
    I=int(f,t,a,b);
    I1(:,j)=I;%alafa与beta內积矩阵
end

B=I1;
%B=zeros(n);%系数矩阵
%for i=1:n
%    for j=1:n
%        B(i,j)=B(i,j)-b^(i+j-1)/(factorial(j-1)*(i+j-1));
%    end
%end
%B
I=ones(n,1);
for i=1:n
    f=y*u^(i-1);k=2;%beta与y的乘积;k为高斯求积的插值点数
    I(i)=int(y*u^(i-1),u,a,b);
    %I(i)=GS_I(k,f,a,b,m);
    I2=I;%y与beta的內积向量
end
A=lambda*eye(n)-B;%求c的系数矩阵
c=A\I2;
