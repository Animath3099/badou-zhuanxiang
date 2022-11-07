import torch.cuda

from loader import load_data



class Evaluate():
    def __init__(self, config, model, logger, knwb):
        self.config = config
        self.model = model
        self.logger = logger
        self.init_knwb(knwb)
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info("开始第%d轮模型测试：" % epoch)

        # 初始化结果集
        self.stats_dict = {"correct": 0, "wrong": 0}

        # 获取此轮测试各个问题对应的向量
        self.knwb_vectors = self.get_knwb_vectors()

        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                question_vectors = self.model(input_ids)
            self.write_status(question_vectors, labels)
        self.show_status()

    def init_knwb(self, knwb):
        self.question_id_to_label_index = {}
        self.question_ids = []
        for target, question_ids in knwb.items():
            for question_id in question_ids:
                self.question_id_to_label_index[len(self.question_ids)] = target
                self.question_ids.append(question_id)
        self.question_matrixs = torch.stack(self.question_ids, dim=0)
        if torch.cuda.is_available():
            self.question_matrixs = self.question_matrixs.cuda()

    def get_knwb_vectors(self):
        with torch.no_grad():
            knwb_vectors = self.model(self.question_matrixs)
            knwb_vectors = torch.nn.functional.normalize(knwb_vectors, dim=-1)
            return knwb_vectors

    def write_status(self, question_vectors, labels):
        for question_vector, label in zip(question_vectors, labels):
            res = torch.mm(question_vector.unsqueeze(0), self.knwb_vectors.T)  # (1,max_length), (max_length,question_nums)
            question_id = int(torch.argmax(res))
            target = self.question_id_to_label_index[question_id]
            if int(target) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def show_status(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测总条目数：%d" % (correct + wrong))
        self.logger.info("预测正确数：%d" % correct)
        self.logger.info("预测错误数：%d" % wrong)
        self.logger.info("预测正确占比：%f" % (correct / (correct + wrong)))
        self.logger.info("-------------------------------")
