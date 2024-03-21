# This file is part of Yolov8-UCMCTrack-DeepSort-MOT which is released under the AGPL-3.0 license.
# See file LICENSE or go to https://github.com/Yangqun123456/Yolov8-UCMCTrack-DeepSort-MOT/tree/main/LICENSE for full license details.

from pyecharts.charts import Line
from pyecharts import options as opts

class GraphData:
    def __init__(self, person, bicycle, car, bus, truck, boat):
        self.person = person # id = 0
        self.bicycle = bicycle # id = 1
        self.car = car # id = 2
        self.bus = bus # id = 5
        self.truck = truck # id = 7
        self.boat = boat # id = 8

def analyzeData(object_ids):
    person = 0
    bicycle = 0
    car = 0
    bus = 0
    truck = 0
    boat = 0
    for object_id in object_ids:
        if object_id == 0:
            person += 1
        elif object_id == 1:
            bicycle += 1
        elif object_id == 2:
            car += 1
        elif object_id == 5:
            bus += 1
        elif object_id == 7:
            truck += 1
        elif object_id == 8:
            boat += 1
    return GraphData(person, bicycle, car, bus, truck, boat)

    
def Scatter(timesListGraph, graphDataList):  # 绘制折线图html文件
    timesListGraph = timesListGraph[-40:]   
    graphDataList = graphDataList[-40:]  

    person_list = [data.person for data in graphDataList]
    bicycle_list = [data.bicycle for data in graphDataList]
    car_list = [data.car for data in graphDataList]
    bus_list = [data.bus for data in graphDataList]
    truck_list = [data.truck for data in graphDataList]
    boat_list = [data.boat for data in graphDataList]

    c = (
        Line(init_opts=opts.InitOpts(
            width="900px",
            height="600px",
            animation_opts=opts.AnimationOpts(animation=False)
        )).add_xaxis(timesListGraph)
        .add_yaxis('人', person_list)
        .add_yaxis('自行车', bicycle_list)
        .add_yaxis('汽车', car_list)
        .add_yaxis('公交车', bus_list)
        .add_yaxis('卡车', truck_list)
        .add_yaxis('船', boat_list)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="实时流量变化"),
            yaxis_opts=opts.AxisOpts(name="数量"),
            xaxis_opts=opts.AxisOpts(name="时间"))
    ).render("output\chart.html")  # 生成折线图html文件
