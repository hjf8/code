load data3 A b m x1
B=A*m;
r=[];
for i=1:10
    r=[r,b-B(:,i)];
end
r
for j=1:10
    y(j)=norm(r(:,j));
    end
y
x=[1:10];
plot(x,y,'-*');
