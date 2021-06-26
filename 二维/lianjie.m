function lianjie = lianjie(node_x,node_y,nel)
%输入横纵坐标的节点数目，和单元自由度
%输出连接矩阵，每个单元涉及的节点的编号
%node_x=5;node_y=5;nel=3;
xn = 1:(node_x*node_y);%编号一列排下来
A = reshape(xn,node_x,node_y);%同形状编号

for i = 1:(node_x-1)*(node_y-1)%矩形单元
    x = rem(i,node_x-1);%表示单元为x方向数起第几个   
    if x == 0
        x = node_x-1;
    end
    y = ceil(i/(node_x-1));%向上取整,y方向数起第几个
    a = A(x:x+1,y:y+1);%这个小矩阵，拉直了就是连接矩阵
    a_vec = a(:);
    lianjie(2*i-1:2*i,1:nel) = [a_vec([1 4 3])';a_vec([4 1 2])'];
end



