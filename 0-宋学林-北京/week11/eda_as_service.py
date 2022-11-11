import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpclient
from tornado.options import define, options
import re
from eda import *


# 根据入参返回
class HelloHandler(tornado.web.RequestHandler):
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')
        self.write(greeting + ', friendly user!')


# 文本增强
class AugHandler(tornado.web.RequestHandler):
    def get(self):
        string = self.get_argument('string')
        sen_list = eda(sentence=string)
        res = ''
        for sen, method in sen_list:
            res += '%s  ||  增强方法：%s <br>' %(sen, method)
        self.write(res)



if __name__ == "__main__":
    define("port", default=1234, help="run on the given port", type=int)

    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/hello", HelloHandler),
                                            (r"/aug", AugHandler),
                                            ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()