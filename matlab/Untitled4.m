load data3 A b m x1
g=[];
for i=1:10
    g=[g,m(:,i)-x1];
end
g%����ÿһ�м�ȥ��ȷ��
for j=1:10
    y(j)=norm(g(:,j));
    end
y
x=[1:10];
plot(x,y,'*');