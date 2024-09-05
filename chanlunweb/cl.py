import datetime
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import talib as ta
from  chanlunweb.cl_interface import *
from  ChanConfig import CChanConfig
from  Chan import CChan
from  Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from  KLine.KLine_Unit import CKLine_Unit
from  Common.CTime import CTime
from  Common.func_util import kltype_lt_day, str2float
from  Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from  KLine.KLine import *
from  Seg.Seg import *
from  ZS.ZS import *
from  Bi import *
from  BuySellPoint.BS_Point import CBS_Point

frequency_maps = {
        "1s": KL_TYPE.K_1S,
        "3s": KL_TYPE.K_3S,
        "5s": KL_TYPE.K_5S,
        "10s": KL_TYPE.K_10S,
        "15s": KL_TYPE.K_15S,
        "20s": KL_TYPE.K_20S,
        "30s": KL_TYPE.K_30S,
        "1m": KL_TYPE.K_1M,
        "3m": KL_TYPE.K_3M,
        "5m": KL_TYPE.K_5M,
        "10m": KL_TYPE.K_10M,
        "15m": KL_TYPE.K_15M,
        "30m": KL_TYPE.K_30M,
        "60m": KL_TYPE.K_60M,
        "d": KL_TYPE.K_DAY,
        
        "w": KL_TYPE.K_WEEK,
        "m": KL_TYPE.K_MON,
        "y": KL_TYPE.K_YEAR,

        "2d": KL_TYPE.K_2DAY,
        "2m": KL_TYPE.K_2M,
        "120m": KL_TYPE.K_120M,
        "3h": KL_TYPE.K_3H,
        "4h": KL_TYPE.K_4H,
    }

           
def GetColumnNameFromFieldList(fileds: str):
    _dict = {
        "time": DATA_FIELD.FIELD_TIME,
        "date": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
        "volume": DATA_FIELD.FIELD_VOLUME,
        "amount": DATA_FIELD.FIELD_TURNOVER,
        "turn": DATA_FIELD.FIELD_TURNRATE,
    }
    return [_dict[x] for x in fileds.split(",")]

def parse_time_column(inp):
    # 20210902113000000
    # 2021-09-13
    if len(inp) == 10:
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = minute = 0
    elif len(inp) == 17:
        year = int(inp[:4])
        month = int(inp[4:6])
        day = int(inp[6:8])
        hour = int(inp[8:10])
        minute = int(inp[10:12])
    elif len(inp) == 19:
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = int(inp[11:13])
        minute = int(inp[14:16])
    else:
        raise Exception(f"unknown time column from baostock:{inp}")
    return CTime(year, month, day, hour, minute)

def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if i == 0 else str2float(data[i])
    return dict(zip(column_name, data))
# 将chan内的合并k线转换成当前合并k线
def klc_to_clk( klc:CKLine, is_up:bool) -> CLKline:
    if klc == None:
        return None
    klu = None
    is_up = klc.dir== KLINE_DIR.UP
    if is_up:
        klu =  klc.get_peak_klu(is_high=True)
    else:
        klu =  klc.get_peak_klu(is_high=False)
    kl = None
    klines:List[Kline] = []
    for k in klc.lst:
        kl = Kline(index=k.idx,date=k.time.toDateTime(),h=k.high,l=k.low,o=k.open,c=k.close,a=k.volume)
        klines.append(kl)
    #合并K线
    return CLKline(k_index=klu.idx,date=klu.time.toDateTime(),h=klu.high,l=klu.low,o=klu.open,c=klu.close,index=klc.idx
                    ,a=klc.get_klu_volume(),klines=klines)    
# 将chan内的合并k线转换成当前合并k线
def klc_to_clk_ck( klc:CKLine) -> CLKline:
    if klc == None:
        return None
    kl = None
    klines:List[Kline] = []
    for k in klc.lst:
        kl = Kline(index=k.idx,date=k.time.toDateTime(),h=k.high,l=k.low,o=k.open,c=k.close,a=k.volume)
        klines.append(kl)
    #合并K线
    return CLKline(k_index=klc.idx,date=klc.time.toDateTime(),h=klc.high,l=klc.low,o=klc.open,c=klc.close,index=klc.idx
                    ,a=klc.get_klu_volume(),klines=klines)    
    # 将chan内的k线转换成当前k线
def klu_to_lk( klu) -> Kline:
    return Kline(index=klu.idx, date=klu.time.toDateTime(),h=klu.high,l=klu.low,o=klu.open,c=klu.close,a=klu.volume)
    
class CL(ICL):
    """
    行情数据缠论分析
    """

    def __init__(self, code: str, frequency: str, config: [dict, None] = None):
        """
        缠论计算
        :param code: 代码
        :param frequency: 周期
        :param config: 配置
        """
        self.code = code
        self.frequency = frequency
        self.config = config if config else {}

        # 是否标识出未完成的笔， True 标识，False 不标识
        if 'no_bi' not in self.config:
            self.config['no_bi'] = True
        # 笔类型 old  老笔  new 新笔  dd 顶底成笔
        if 'bi_type' not in self.config:
            self.config['bi_type'] = 'old'
        # 分型是否允许包含成笔的配置，False 不允许分型包含成笔，True 允许分型包含成笔
        if 'fx_baohan' not in self.config:
            self.config['fx_baohan'] = False
        # 中枢类型，dn 段内中枢，以线段为主，只算线段内的中枢，bl 遍历查找所有存在的中枢
        if 'zs_type' not in self.config:
            self.config['zs_type'] = 'dn'
        # 中枢标准，hl 实际高低点，dd 顶底端点
        if 'zs_qj' not in self.config:
            self.config['zs_qj'] = 'hl'

        # 指标配置项
        idx_keys = {
            'idx_macd_fast': 12,
            'idx_macd_slow': 26,
            'idx_macd_signal': 9,
            'idx_boll_period': 20,
            'idx_ma_period': 5,
        }
        for _k in idx_keys.keys():
            if _k not in self.config:
                self.config[_k] = idx_keys[_k]
            else:
                self.config[_k] = int(self.config[_k])

        # 计算后保存的值
        self.klines: List[Kline] = []  # 整理后的原始K线
        self.cl_klines: List[CLKline] = []  # 缠论K线
        self.idx: dict = {
            'macd': {'dea': [], 'dif': [], 'hist': []},
            'ma': [],
            'boll': {'up': [], 'mid': [], 'low': []},
        }  # 各种行情指标
      
        config = CChanConfig({
                    "bi_strict": True,
                    "trigger_step": True,
                    "skip_step": 0,
                    "divergence_rate": 0.9,
                    "bsp2_follow_1": False,
                    "bsp3_follow_1": False,
                    "min_zs_cnt": 1,
                    # "bs1_peak": False,
                    "macd_algo": "slope",
                    # "bs_type": '1,2,3a,1p,2s,3b',
                    "print_warning": True,
                    "zs_algo": "normal",
                })
        lv_list = [frequency_maps[self.frequency]]
        chan_t = CChan(
                code=self.code,
                lv_list=lv_list,
                config=config,
                )

        self.chan: CChan = chan_t
        self.use_time = {}  # debug 调试用时

    def get_bi_zss(self, zs_type: str = None) -> List[ZS]:
        zss : List[ZS] = []
        for k in self.chan[0].zs_list.zs_lst:
            zs = self.chan_zs_to_cl(k,'bi')
            if zs:
                zss.append(zs)
        return zss
    def get_bis(self) -> List[BI]:
        bis: List[BI] =[]
        for k in self.chan[0].bi_list:
            bi = self.chan_bi_to_cl(k)
            if bi:
                bis.append(bi)
        return bis
    def get_cl_klines(self) -> List[CLKline]:
        cks : List[CLKline] = []
        for k in self.chan.kl_datas[0].lst:
            ck =klc_to_clk_ck(k)
            if ck:
                cks.append(ck)
        return cks
    def get_bi_mmds(self) -> List[CBS_Point]:
        return self.chan.get_bsp()
    def get_xd_mmds(self) -> List[CBS_Point]:
        return sorted(self.chan[0].seg_bs_point_lst.lst, key=lambda x: x.klu.time)
    def get_code(self) -> str:
        return self.code
    def get_frequency(self) -> str:
        return self.frequency
    def get_fxs(self) -> List[FX]:
        return []
    def get_idx(self) -> dict:
        return self.idx
    def get_klines(self) -> List[Kline]:
        return self.klines
    def get_last_bi_zs(self) -> List[ZS]:
        return []
    def get_last_xd_zs(self) -> List[ZS]:
        return []
    def get_qsd_zss(self) -> List[ZS]:
        return []
    def get_qsds(self)  -> List[XD]:
        return []
    def get_src_klines(self) -> List[Kline]:
        return self.klines
    def get_xd_zss(self, zs_type: str = None) -> List[ZS]:
        zss : List[ZS] = []
        for k in self.chan[0].segzs_list.zs_lst:
            zs = self.chan_zs_to_cl(k,'xd')
            if zs:
                zss.append(zs)
        return zss
    def get_xds(self) -> List[XD]:
        xds: List[XD] =[]
        for k in self.chan[0].seg_list:
            xd = self.chan_xd_to_cl(k)
            if xd:
                xds.append(xd)
        return xds
    def get_zsd_zss(self) -> List[ZS]:
        return []
    def get_zsds(self) -> List[XD]:
        """
        返回计算缠论走势段列表
        """
        return []

    def zss_is_qs(self, one_zs: ZS, two_zs: ZS) -> Tuple[str, None]:
        """
        判断两个中枢是否形成趋势（根据设置的位置关系配置，来判断两个中枢是否有重叠）
        返回  up 是向上趋势， down 是向下趋势 ，None 则没有趋势
        """
        if one_zs.dd > two_zs.gg:
            return "down"
        if one_zs.gg < two_zs.dd:
            return "up"
        return None
    def __add_time(self, key, use_time):
        """
        debug 增加用时
        """
        if key not in self.use_time:
            self.use_time[key] = use_time
        else:
            self.use_time[key] += use_time
 
    def process_klines(self, klines: pd.DataFrame):
        """
        计算k线缠论数据
        传递 pandas 数据，需要包括以下列：
            date  时间日期  datetime 格式
            high  最高价
            low   最低价
            open  开盘价
            close  收盘价
            volume  成交量

        可增量多次调用，重复已计算的会自动跳过，最后一个 bar 会进行更新
        """
        k_index = len(self.klines)
        for _k in klines.iterrows():
            k = _k[1]
            if len(self.klines) == 0:
                nk = Kline(index=k_index, date=k['date'], h=float(k['high']), l=float(k['low']),
                           o=float(k['open']), c=float(k['close']), a=float(k['volume']))
                self.klines.append(nk)
                k_index += 1
                continue
            if self.klines[-1].date > k['date']:
                continue
            if self.klines[-1].date == k['date']:
                self.klines[-1].h = float(k['high'])
                self.klines[-1].l = float(k['low'])
                self.klines[-1].o = float(k['open'])
                self.klines[-1].c = float(k['close'])
                self.klines[-1].a = float(k['volume'])
            else:
                nk = Kline(index=k_index, date=k['date'], h=float(k['high']), l=float(k['low']),
                           o=float(k['open']), c=float(k['close']), a=float(k['volume']))
                self.klines.append(nk)
                k_index += 1
                ckud = [
                 k['date'].strftime('%Y-%m-%d %H:%M:%S'), k['open'],
                 k['high'], k['low'],
                k['close'], k['volume']
                ]
                fields = "date,open,high,low,close,volume,amount,turn"
                cku = CKLine_Unit(create_item_dict(ckud, GetColumnNameFromFieldList(fields)))
                cku.set_idx(k_index)
                self.chan.trigger_load({frequency_maps[self.frequency]: [cku]})  # 喂给CChan新增k线
            # 处理指标
            _s = time.time()
            self.process_idx()
            self.__add_time('process_idx', time.time() - _s)
            
            
            # nk = Kline(index=k_index, date=k['date'], h=float(k['high']), l=float(k['low']),
            #                o=float(k['open']), c=float(k['close']), a=float(k['volume']))
        return self

    def process_fx_chan(self):
        pass

    # zs_type: str = zs_type  # 标记中枢类型 bi 笔中枢 xd 线段中枢 zsd 走势段中枢
    def chan_zs_to_cl(self,zs :CZS,zs_type: str) -> ZS:
        bi_be :CBi
        bi_en :CBi
        if zs_type == 'bi':
            bi_be = zs.begin_bi
            bi_en = zs.end_bi
        elif zs_type == 'xd':
            bi_be = zs.begin_bi.start_bi
            bi_en = zs.end_bi.end_bi
        if bi_en == None :
            return None
        beg_klc = bi_be.begin_klc
        end_klc = bi_en.end_klc
        is_up = bi_be.is_up()
        sfx = FX(index=1, _type='ding' if beg_klc.fx == FX_TYPE.TOP else 'di', k= klc_to_clk(beg_klc,is_up), klines=[klc_to_clk(beg_klc.pre,is_up), klc_to_clk(beg_klc,is_up), klc_to_clk(beg_klc.next,is_up)], 
                    val=beg_klc.high if beg_klc.fx == FX_TYPE.TOP else beg_klc.low,
                done=bi_be.is_sure)
        is_up = bi_en.is_up()
        efx = FX(index=1, _type='ding' if end_klc.fx == FX_TYPE.TOP else 'di', k= klc_to_clk(end_klc,is_up), klines=[klc_to_clk(end_klc.pre,is_up), klc_to_clk(end_klc,is_up), klc_to_clk(end_klc.next,is_up)], 
                    val=end_klc.high if end_klc.fx == FX_TYPE.TOP else end_klc.low,
                done=bi_en.is_sure)
        newzs = ZS(zs_type=zs_type,start=sfx,end=efx,zg=zs.high,zd=zs.low,gg=zs.peak_high,dd=zs.peak_low,_type= 'up' if beg_klc.fx == FX_TYPE.TOP else 'down',level=0)
        newzs.done = zs.is_sure
        return newzs

    def chan_bi_to_cl(self,bi :CBi) -> BI:
        begin_klc = bi.begin_klc
        end_klc = bi.end_klc
        is_up = bi.is_up()
        klus = bi.get_begin_klu()
        klue = bi.get_end_klu()
        # if is_up:
        #     klu =  begin_klc.get_peak_klu(is_high=False)
        #     klue =  end_klc.get_peak_klu(is_high=True)
        # else:
        #     klu =  begin_klc.get_peak_klu(is_high=True)
        #     klue =  end_klc.get_peak_klu(is_high=False)
        lks = klu_to_lk(klus)
        lke = klu_to_lk(klue)
         # 将chan笔转换成当前笔
        sfx = FX(index=1, _type='ding' if begin_klc.fx == FX_TYPE.TOP else 'di', k= klc_to_clk(begin_klc,is_up), klines=[klc_to_clk(begin_klc.pre,is_up), klc_to_clk(begin_klc,is_up), klc_to_clk(begin_klc.next,is_up)], 
                    val=lks.h if begin_klc.fx == FX_TYPE.TOP else lks.l,
                done=bi.is_sure)
        new_bi = None
        if bi.is_sure:
            efx = FX(index=1, _type='ding' if end_klc.fx == FX_TYPE.TOP else 'di', k= klc_to_clk(end_klc,is_up), klines=[klc_to_clk(end_klc.pre,is_up), klc_to_clk(end_klc,is_up), klc_to_clk(end_klc.next,is_up)], 
                        val=lke.h if end_klc.fx == FX_TYPE.TOP else lke.l,
                    done=bi.is_sure)
            # 新笔产生了
            new_bi = BI(start=sfx, end=efx)
            new_bi.index = bi.idx
            
            new_bi.type = 'up' if bi.is_up() else 'down'
            new_bi.fx_num = new_bi.end.index - new_bi.start.index
            self.process_line_ld(new_bi)  # 计算笔力度
            self.process_line_hl(new_bi)  # 计算实际高低点
            
        elif self.config['no_bi'] and end_klc!= None:
            efx = FX(index=1, _type='ding' if end_klc.fx == FX_TYPE.TOP else 'di', k=  klc_to_clk(end_klc,is_up), klines=[klc_to_clk(end_klc.pre,is_up), klc_to_clk(end_klc,is_up), klc_to_clk(end_klc.next,is_up)], 
                        val=lke.h if end_klc.fx == FX_TYPE.TOP else lke.l,done=False)
            # 新笔产生了
            new_bi = BI(start=sfx, end=efx)
            new_bi.index = bi.idx
            new_bi.type = 'up' if bi.is_up() else 'down'
            new_bi.fx_num = new_bi.end.index - new_bi.start.index
            self.process_line_ld(new_bi)  # 计算笔力度
            self.process_line_hl(new_bi)  # 计算实际高低点
        #添加买卖点
        if new_bi :
            
            cmmd : CBS_Point  = bi.bsp
            
            # new_bi.add_mmd(self.chan_mmd_to(cmmd))
        return new_bi
    # 添加买卖点
    def chan_mmd_to(self,cbp : CBS_Point) -> MMD:
        print(cbp)
        pass
    def chan_xd_to_cl(self,seg :CSeg) -> XD:
        sbi = self.chan_bi_to_cl(seg.start_bi)
        ebi = self.chan_bi_to_cl(seg.end_bi)
        newXd = XD(sbi.start,ebi.end,sbi,ebi,sbi.type,index=seg.idx)
        newXd.done = seg.is_sure
        self.process_line_ld(newXd)
        self.process_line_hl(newXd)
        return newXd

    def process_bi_chan(self):
        cbi_lst = self.chan[0].bi_list
        for bi in cbi_lst:
            # 起始K线
            begin_klc = bi.begin_klc
            end_klc = bi.end_klc
            is_up = bi.is_up()
            cl_k = klc_to_clk(begin_klc, is_up)
            self.cl_klines.append(cl_k)

            klu = None
            klue = None
            if is_up:
                klu =  begin_klc.get_peak_klu(is_high=False)
                klue =  end_klc.get_peak_klu(is_high=True)
            else:
                klu =  begin_klc.get_peak_klu(is_high=True)
                klue =  end_klc.get_peak_klu(is_high=False)
            lk = klu_to_lk(klu)
            lke = klu_to_lk(klue)
            # 将chan笔转换成当前笔
            sfx = FX(index=len(self.fxs)+1, _type='ding' if begin_klc.fx == FX_TYPE.TOP else 'di', k= klc_to_clk(begin_klc,is_up), klines=[klc_to_clk(begin_klc.pre,is_up), klc_to_clk(begin_klc,is_up), klc_to_clk(begin_klc.next,is_up)], 
                     val=lk.h if begin_klc.fx == FX_TYPE.TOP else lk.l,
                    done=bi.is_sure)
            self.fxs.append(sfx)
            if bi.is_sure:
                efx = FX(index=len(self.fxs)+1, _type='ding' if end_klc.fx == FX_TYPE.TOP else 'di', k= klc_to_clk(end_klc,is_up), klines=[klc_to_clk(end_klc.pre,is_up), klc_to_clk(end_klc,is_up), klc_to_clk(end_klc.next,is_up)], 
                            val=lke.h if end_klc.fx == FX_TYPE.TOP else lke.l,
                        done=bi.is_sure)
                self.fxs.append(efx)
                if len(self.bis) == 0:
                    # 新笔产生了
                    new_bi = BI(start=sfx, end=efx)
                    new_bi.index = len(self.bis) + 1
                    new_bi.type = 'up' if bi.is_up() else 'down'
                    new_bi.fx_num = new_bi.end.index - new_bi.start.index
                    self.process_line_ld(new_bi)  # 计算笔力度
                    self.process_line_hl(new_bi)  # 计算实际高低点
                    self.bis.append(new_bi)
                elif self.bis[-1].end :
                    # 新笔产生了
                    self.bis[-1].end = efx
                    self.bis[-1].fx_num = new_bi.end.index - new_bi.start.index
                    self.process_line_ld(self.bis[-1])  # 计算笔力度
                    self.process_line_hl(self.bis[-1])  # 计算实际高低点
            else:
                # 新笔产生了
                new_bi = BI(start=sfx, end=sfx)
                new_bi.index = len(self.bis) + 1
                new_bi.type = 'up' if bi.is_up() else 'down'
                self.bis.append(new_bi)

    def process_idx(self):
        """
        计算指标
        """
        if len(self.klines) < 2:
            return False

        # 根据指标设置的参数，取最大值，用来获取要计算的价格序列
        idx_keys = [
            'idx_macd_fast',
            'idx_macd_slow',
            'idx_macd_signal',
            'idx_boll_period',
            'idx_ma_period',
        ]
        p_len = 0
        for _k in idx_keys:
            p_len = max(p_len, self.config[_k])

        prices = [k.c for k in self.klines[-(p_len + 10):]]
        # 计算 macd
        macd_dif, macd_dea, macd_hist = ta.MACD(np.array(prices),
                                                fastperiod=self.config['idx_macd_fast'],
                                                slowperiod=self.config['idx_macd_slow'],
                                                signalperiod=self.config['idx_macd_signal'])
        # macd = {'dea': macd_dea, 'dif': macd_dif, 'hist': macd_hist}

        # 计算 ma
        ma = ta.MA(np.array(prices), timeperiod=self.config['idx_ma_period'])

        # 计算 BOLL 指标
        boll_up, boll_mid, boll_low = ta.BBANDS(np.array(prices), timeperiod=self.config['idx_boll_period'])

        for i in range(2, 0, -1):
            index = self.klines[-i].index
            if (len(self.idx['ma']) - 1) >= index:
                # 更新操作
                self.idx['ma'][index] = ma[-i]
                self.idx['macd']['dea'][index] = macd_dea[-i]
                self.idx['macd']['dif'][index] = macd_dif[-i]
                self.idx['macd']['hist'][index] = macd_hist[-i]
                self.idx['boll']['up'][index] = boll_up[-i]
                self.idx['boll']['mid'][index] = boll_mid[-i]
                self.idx['boll']['low'][index] = boll_low[-i]
            else:
                # 添加操作
                self.idx['ma'].append(ma[-i])
                self.idx['macd']['dea'].append(macd_dea[-i])
                self.idx['macd']['dif'].append(macd_dif[-i])
                self.idx['macd']['hist'].append(macd_hist[-i])
                self.idx['boll']['up'].append(boll_up[-i])
                self.idx['boll']['mid'].append(boll_mid[-i])
                self.idx['boll']['low'].append(boll_low[-i])

        # self.idx = {
        #     'macd': macd,
        #     'ma': ma,
        #     'boll': {'up': boll_up, 'mid': boll_mid, 'low': boll_low}
        # }
        return True

    def beichi_line(self, pre_line: LINE, now_line: LINE):
        """
        计算两线之间是否有背驰，两笔必须是同方向的，最新的线创最高最低记录
        背驰 返回 True，否则返回 False
        """
        if pre_line.type != now_line.type:
            return False
        if pre_line.type == 'up' and now_line.high < pre_line.high:
            return False
        if pre_line.type == 'down' and now_line.low > pre_line.low:
            return False

        return self.compare_ld_beichi(pre_line.ld, now_line.ld)

    def beichi_pz(self, zs: ZS, now_line: LINE):
        """
        判断中枢是否有盘整背驰，中枢最后一线要创最高最低才可比较

        """
        if zs.lines[-1].index != now_line.index:
            return False
        if zs.type not in ['up', 'down']:
            return False

        return self.compare_ld_beichi(zs.lines[0].ld, now_line.ld)

    def beichi_qs(self, zss: List[ZS], zs: ZS, now_line: LINE):
        """
        判断是否是趋势背驰，首先需要看之前是否有不重合的同向中枢，在进行背驰判断
        """
        if zs.type not in ['up', 'down']:
            return False

        # 查找之前是否有同向的，并且级别相同的中枢
        pre_zs = [
            _zs for _zs in zss
            if (_zs.lines[-1].index == zs.lines[0].index and _zs.type == zs.type and _zs.level == zs.level)
        ]
        if len(pre_zs) == 0:
            return False
        # 判断 高低点是否有重合
        pre_ok_zs = []
        for _zs in pre_zs:
            if (_zs.type == 'up' and _zs.gg < zs.dd) or (_zs.type == 'down' and _zs.dd > zs.gg):
                pre_ok_zs.append(_zs)

        if len(pre_ok_zs) == 0:
            return False

        return self.compare_ld_beichi(zs.lines[0].ld, now_line.ld)

    def create_zs(self, zs_type: str, zs: [ZS, None], lines: List[LINE]) -> [ZS, None]:
        """
        根据线，获取是否有共同的中枢区间
        zs_type 标记中枢类型（笔中枢 or 线段中枢）
        lines 中，第一线是进入中枢的，不计算高低，最后一线不一定是最后一个出中枢的，如果是最后一个出中枢的，则不需要计算高低点
        """
        if len(lines) <= 3:
            return None

        # 进入段要比中枢第一段高或低
        # if lines[0].type == 'up' and lines[0].low > lines[1].low:
        #     return None
        # if lines[0].type == 'down' and lines[0].high < lines[1].high:
        #     return None
        def line_hl(_l: LINE):
            if self.config['zs_qj'] == 'hl':
                return [_l.high, _l.low]
            else:
                return [_l.ding_high(), _l.di_low()]

        run_lines = []
        zs_done = False
        # 记录重叠线的最高最低值
        _l_one_hl = line_hl(lines[0])
        _high = _l_one_hl[0]
        _low = _l_one_hl[1]
        cross_qj = self.cross_qujian(line_hl(lines[1]), line_hl(lines[3]))
        if cross_qj is None:
            return None
        # 获取所有与中枢有重叠的线，计算中枢，只要有不重叠的，即为中枢结束
        for _l in lines:
            _l_hl = line_hl(_l)
            if self.cross_qujian([cross_qj.line.ding_high, cross_qj.line.di_low], _l_hl):
                _high = max(_high, _l_hl[0])
                _low = min(_low, _l_hl[1])
                run_lines.append(_l)
            else:
                zs_done = True
                break
        # 看看中枢笔数是否足够
        if len(run_lines) < 4:
            return None

        # 如果最后一笔向上并且是最高的，或者最后一笔向下是最低点，则最后一笔不划归中枢，默认为中枢结束一笔，当然后续会扩展
        _last_line = run_lines[-1]
        _last_hl = line_hl(_last_line)
        last_line_in_zs = True
        if (_last_line.type == 'up' and _last_hl[0] == _high) \
                or (_last_line.type == 'down' and _last_hl[1] == _low):
            last_line_in_zs = False

        if zs is None:
            zs = ZS(zs_type=zs_type, start=run_lines[1].start, _type='zd')
        zs.done = zs_done

        zs.lines = []
        zs.add_line(run_lines[0])

        zs_fanwei = [cross_qj.line.ding_high, cross_qj.line.di_low]
        zs_gg = run_lines[1].high
        zs_dd = run_lines[1].low

        for i in range(1, len(run_lines)):
            # 当前线的交叉范围
            _l = run_lines[i]
            _l_hl = line_hl(_l)
            cross_fanwei = self.cross_qujian(zs_fanwei, _l_hl)
            if cross_fanwei is None:
                raise Exception('中枢不可有不交叉的地方')

            if i == len(run_lines) - 1 and last_line_in_zs is False:
                # 判断是最后一线，并且最后一线不在中枢里
                pass
            else:
                zs_gg = max(zs_gg, _l_hl[0])
                zs_dd = min(zs_dd, _l_hl[1])
                # 根据笔数量，计算级别
                zs.line_num = len(zs.lines) - 1
                zs.level = int(zs.line_num / 9)
                zs.end = _l.end
                # 记录中枢中，最大的笔力度
                if zs.max_ld is None:
                    zs.max_ld = _l.ld
                elif _l.ld:
                    zs.max_ld = zs.max_ld if self.compare_ld_beichi(zs.max_ld, _l.ld) else _l.ld
            zs.add_line(_l)

        zs.zg = zs_fanwei[0]
        zs.zd = zs_fanwei[1]
        zs.gg = zs_gg
        zs.dd = zs_dd

        # 计算中枢方向
        if zs.lines[0].type == zs.lines[-1].type:
            _l_start_hl = line_hl(zs.lines[0])
            _l_end_hl = line_hl(zs.lines[-1])
            if zs.lines[0].type == 'up' and _l_start_hl[1] <= zs.dd and _l_end_hl[0] >= zs.gg:
                zs.type = zs.lines[0].type
            elif zs.lines[0].type == 'down' and _l_start_hl[0] >= zs.gg and _l_end_hl[1] <= zs.dd:
                zs.type = zs.lines[0].type
            else:
                zs.type = 'zd'
        else:
            zs.type = 'zd'

        return zs

    def create_dn_zs(self, zs_type: str, lines: List[LINE]) -> List[ZS]:
        """
        计算端内同向中枢
        """
        zss: List[ZS] = []
        if len(lines) <= 4:
            return zss

        start = 0
        while True:
            run_lines = lines[start:]
            if len(run_lines) == 0:
                break
            zs = self.create_zs(zs_type, None, run_lines)
            if zs is None:
                start += 1
            else:
                zss.append(zs)
                start += len(zs.lines) - 1

        return zss

    def process_line_ld(self, line: LINE):
        """
        处理并计算线（笔、线段）的力度
        """
        line.ld = {
            'macd': self.query_macd_ld(line.start, line.end)
        }
        return True

    def process_line_hl(self, line: LINE):
        """
        处理并计算线（笔、线段）实际高低点
        """
        hl = self.__fx_qj_high_low(line.start, line.end)
        line.high = hl['high']
        line.low = hl['low']
        return True

    def query_macd_ld(self, start_fx: FX, end_fx: FX):
        """
        计算分型区间 macd 力度
        """
        if start_fx.index > end_fx.index:
            raise Exception('%s - %s - %s 计算力度，开始分型不可以大于结束分型' % (self.code, self.frequency, self.klines[-1].date))

        dea = np.array(self.idx['macd']['dea'][start_fx.k.k_index:end_fx.k.k_index + 1])
        dif = np.array(self.idx['macd']['dif'][start_fx.k.k_index:end_fx.k.k_index + 1])
        hist = np.array(self.idx['macd']['hist'][start_fx.k.k_index:end_fx.k.k_index + 1])
        if len(hist) == 0:
            hist = np.array([0])
        if len(dea) == 0:
            dea = np.array([0])
        if len(dif) == 0:
            dif = np.array([0])

        hist_abs = abs(hist)
        hist_up = np.array([_i for _i in hist if _i > 0])
        hist_down = np.array([_i for _i in hist if _i < 0])
        hist_sum = hist_abs.sum()
        hist_up_sum = hist_up.sum()
        hist_down_sum = hist_down.sum()
        end_dea = dea[-1]
        end_dif = dif[-1]
        end_hist = hist[-1]
        return {
            'dea': {'end': end_dea, 'max': np.max(dea), 'min': np.min(dea)},
            'dif': {'end': end_dif, 'max': np.max(dif), 'min': np.min(dif)},
            'hist': {'sum': hist_sum, 'up_sum': hist_up_sum, 'down_sum': hist_down_sum, 'end': end_hist},
        }

    def __fx_qj_high_low(self, start: FX, end: FX):
        """
        获取分型区间实际的 高低点
        """
        klines = self.klines[start.k.k_index:end.k.k_index + 1]
        k_h = [_k.h for _k in klines]
        k_l = [_k.l for _k in klines]
        high = np.array(k_h).max()
        low = np.array(k_l).min()
        return {'high': high, 'low': low}

    @staticmethod
    def __copy_zs(copy_zs: ZS, to_zs: ZS):
        """
        复制一个中枢的属性到另外一个中枢
        """
        to_zs.zs_type = copy_zs.zs_type
        to_zs.start = copy_zs.start
        to_zs.lines = copy_zs.lines
        to_zs.end = copy_zs.end
        to_zs.zg = copy_zs.zg
        to_zs.zd = copy_zs.zd
        to_zs.gg = copy_zs.gg
        to_zs.dd = copy_zs.dd
        to_zs.type = copy_zs.type
        to_zs.line_num = copy_zs.line_num
        to_zs.level = copy_zs.level
        to_zs.max_ld = copy_zs.max_ld
        to_zs.done = copy_zs.done
        to_zs.real = copy_zs.real
        return

    @staticmethod
    def cross_qujian(qj_one, qj_two):
        """
        计算两个范围相交部分区间
        :param qj_one:
        :param qj_two:
        :return:
        """
        # 判断线段是否与范围值内有相交
        max_one = max(qj_one[0], qj_one[1])
        min_one = min(qj_one[0], qj_one[1])
        max_two = max(qj_two[0], qj_two[1])
        min_two = min(qj_two[0], qj_two[1])

        cross_max_val = min(max_two, max_one)
        cross_min_val = max(min_two, min_one)

        if cross_max_val >= cross_min_val:
            return {'max': cross_max_val, 'min': cross_min_val}
        else:
            return None

def batch_cls(code, klines: Dict[str, pd.DataFrame], config: dict = None) -> List[CL]:
    """
    批量计算并获取 缠论 数据
    :param code:
    :param klines:
    :param config: 缠论配置
    :return:
    """
    cls = []
    for f in klines.keys():
        cls.append(CL(code, f, config).process_klines(klines[f]))
    return cls