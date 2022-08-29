week02作业内容：将Demo中二分类任务改成多分类任务。

自己写作业时一些要点：
<ol>
  <li>修改TorchModule属性，self.classify线性层shape由1维变更至3维，以便进行3分类任务；</li>
  <li>修改TorchModule属性，self.loss = nn.CrossEntropyLoss(),同时注释掉self.activation = torch.sigmoid，因为pytorch交叉熵损失函数内嵌softmax激活函数；</li>
  <li>修改TorchModule前馈函数forward，注释掉y_pred = self.activation(x)因为已经不需要；</li>
  <li>修改TorchModule前馈函数forward，在训练分支第一行添加y = y.squeeze(dim=1).to(torch.int64)，将标签shape变更为(20,)，dtype变更为Long(int64)避免计算损失报错；</li>
  <li>修改函数build sample，变成3类标签：正样本1、正样本2和负样本</li>
  <li>修改函数evaluate，对三个类别的正确率分别计算；</li>
  <li>修改函数main()的画图部分，添加图标题、x坐标轴题和y坐标轴题。</li>
</ol>


作业提交文件目录：
<ol>
  <li>【homework.py】 基于Demo.py修改完成后的代码</li>
  <li>【loss & acc plot.png】Loss & Acc随epoch变化图；</li>
  <li>【train and predict log.md】记录训练和预测输出日志；</li>
</ol>



P.S. 一紧张文件夹名字写错了，应该是week02，但是改起来太麻烦，就这样吧~
