'''
Created on 2017. 8. 2.

@author: 3F8VJ32
'''
import pymysql
import numpy as np
import copy
import json
class JsonParsorClass(object):
    
    sqlConstant = "SELECT * FROM tb_janggi_constant";
    sqlJanggiGame = "SELECT * FROM tb_janggi_game where jg_winner != ''"
    sqlJanggiPan ="SELECT * FROM tb_janggi_pan where jg_idx = %s"
    
    def __init__(self, host='localhost', port=3306, user='ml_user', passwd='mleoqkr!@', db='tensorflow', charset='utf8'):
        self.constantDic = {};
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.db = db
        self.charset=charset
        self.initConstantDic()
    
    def initConstantDic(self):
        cur = self._selectQuery(self.sqlConstant);
        dic = {};
        for row in cur:
            dic.__setitem__(row[0], row[1])
        self.constantDic = dic;
    
    def getGameList(self):
        cur = self._selectQuery(self.sqlJanggiGame)
        rt = []
        
        for row in cur:
            rt.append({"idx": row[0], "title":row[1], "win": row[4], "turn": row[5]})
        return rt
    
    def getPanList(self, game_idx):
        cur = self._selectQuery(self.sqlJanggiPan, game_idx)
        rt = []
        for row in cur:
            win = self.constantDic.get(row[3])
            turn = row[4]
            state = self._parseJson(json.loads(row[2]))
            rt.append({"win":win, "turn":turn, "map": state})
        return rt
    def _getConn(self):

        return pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd, db=self.db, charset=self.charset)
    
    def _selectQuery(self, sql, opt = None):
        conn = self._getConn();
        cur = conn.cursor();
        if opt == None:
            cur.execute(sql)
        else: 
            cur.execute(sql, opt)
        cur.close();
        conn.close();
        return cur;
        
    def _parseJson(self, JsonParsor):
        rt = np.zeros((10, 9, 2))
        for i in range(len(JsonParsor)):
            for j in range(len(JsonParsor[i])):
                if JsonParsor[i][j] == 0:
                    continue;
                # HAN
                if "R_" in JsonParsor[i][j]:
                    rt[i][j][1] = self.constantDic.get(JsonParsor[i][j])
                elif "B_" in JsonParsor[i][j]:
                    rt[i][j][0] = self.constantDic.get(JsonParsor[i][j])
        return rt;
        
        