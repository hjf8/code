function c=middle(n,a,b,lambda)%nΪ�˺����Ľ��ƽ���,mΪ�����ʷ���
%a=0;b=1;lambda=5;n=5;
syms u t
y=const_y(lambda,a,b);
y=subs(y,t,u);
I1=[];
for j=1:n
    i=1:n;i=i';
    f=t^(j-1)/factorial(j-1)*t.^(i-1);%alafa��beta�˻�
    I=int(f,t,a,b);
    I1(:,j)=I;%alafa��beta�Ȼ�����
end

B=I1;
%B=zeros(n);%ϵ������
%for i=1:n
%    for j=1:n
%        B(i,j)=B(i,j)-b^(i+j-1)/(factorial(j-1)*(i+j-1));
%    end
%end
%B
I=ones(n,1);
for i=1:n
    f=y*u^(i-1);k=2;%beta��y�ĳ˻�;kΪ��˹����Ĳ�ֵ����
    I(i)=int(y*u^(i-1),u,a,b);
    %I(i)=GS_I(k,f,a,b,m);
    I2=I;%y��beta�ăȻ�����
end
A=lambda*eye(n)-B;%��c��ϵ������
c=A\I2;
