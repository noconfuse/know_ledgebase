#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据生成器
基于消防法文档生成多样化的测试查询集合
"""

import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class TestCase:
    """测试用例数据结构"""
    id: str
    query: str
    expected_keywords: List[str]
    expected_articles: List[str]
    category: str
    difficulty: str
    query_type: str  # 查询类型：定义、职责、程序、处罚等
    description: str  # 测试用例描述

class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self):
        self.test_cases = []
        self._generate_all_test_cases()
    
    def _generate_all_test_cases(self):
        """生成所有测试用例"""
        # 基础定义类查询
        self._generate_definition_queries()
        
        # 职责义务类查询
        self._generate_responsibility_queries()
        
        # 程序流程类查询
        self._generate_procedure_queries()
        
        # 法律责任类查询
        self._generate_penalty_queries()
        
        # 特殊场所类查询
        self._generate_special_place_queries()
        
        # 组织机构类查询
        self._generate_organization_queries()
        
        # 复合查询
        self._generate_complex_queries()
        
        # 边界情况查询
        self._generate_edge_case_queries()
    
    def _generate_definition_queries(self):
        """生成定义类查询"""
        cases = [
            TestCase(
                id="def_001",
                query="什么是消防设施？",
                expected_keywords=["火灾自动报警系统", "应急广播", "应急照明", "安全疏散设施"],
                expected_articles=["第七十三条"],
                category="基础定义",
                difficulty="easy",
                query_type="定义",
                description="测试对消防设施定义的理解"
            ),
            TestCase(
                id="def_002",
                query="消防产品包括哪些？",
                expected_keywords=["火灾预防", "灭火救援", "火灾防护", "避难", "逃生"],
                expected_articles=["第七十三条"],
                category="基础定义",
                difficulty="easy",
                query_type="定义",
                description="测试对消防产品定义的理解"
            ),
            TestCase(
                id="def_003",
                query="人员密集场所是指什么？",
                expected_keywords=["公众聚集场所", "医院", "学校", "幼儿园", "养老院", "宿舍"],
                expected_articles=["第七十三条"],
                category="基础定义",
                difficulty="medium",
                query_type="定义",
                description="测试对人员密集场所定义的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_responsibility_queries(self):
        """生成职责义务类查询"""
        cases = [
            TestCase(
                id="resp_001",
                query="政府在消防工作中的职责是什么？",
                expected_keywords=["统一领导", "国民经济", "社会发展计划", "经济社会发展相适应"],
                expected_articles=["第三条"],
                category="政府职责",
                difficulty="medium",
                query_type="职责",
                description="测试对政府消防职责的理解"
            ),
            TestCase(
                id="resp_002",
                query="应急管理部门的监督管理职责有哪些？",
                expected_keywords=["监督管理", "消防救援机构", "军事设施", "海上石油天然气设施"],
                expected_articles=["第四条"],
                category="部门职责",
                difficulty="medium",
                query_type="职责",
                description="测试对应急管理部门职责的理解"
            ),
            TestCase(
                id="resp_003",
                query="公民在消防工作中有什么义务？",
                expected_keywords=["维护消防安全", "参加有组织的灭火工作", "成年人"],
                expected_articles=["第五条"],
                category="公民义务",
                difficulty="easy",
                query_type="义务",
                description="测试对公民消防义务的理解"
            ),
            TestCase(
                id="resp_004",
                query="物业服务企业的消防职责是什么？",
                expected_keywords=["共用消防设施", "维护管理", "消防安全防范服务", "住宅区"],
                expected_articles=["第十八条"],
                category="企业职责",
                difficulty="medium",
                query_type="职责",
                description="测试对物业企业消防职责的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_procedure_queries(self):
        """生成程序流程类查询"""
        cases = [
            TestCase(
                id="proc_001",
                query="建设工程消防设计审查的流程是什么？",
                expected_keywords=["特殊建设工程", "消防设计文件", "住房和城乡建设主管部门", "审查"],
                expected_articles=["第十一条"],
                category="审查程序",
                difficulty="hard",
                query_type="程序",
                description="测试对消防设计审查程序的理解"
            ),
            TestCase(
                id="proc_002",
                query="消防验收的程序是怎样的？",
                expected_keywords=["建设工程竣工", "申请消防验收", "备案", "抽查"],
                expected_articles=["第十三条"],
                category="验收程序",
                difficulty="hard",
                query_type="程序",
                description="测试对消防验收程序的理解"
            ),
            TestCase(
                id="proc_003",
                query="公众聚集场所消防安全检查的程序是什么？",
                expected_keywords=["告知承诺管理", "申请", "承诺", "材料", "核查"],
                expected_articles=["第十五条"],
                category="检查程序",
                difficulty="hard",
                query_type="程序",
                description="测试对消防安全检查程序的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_penalty_queries(self):
        """生成法律责任类查询"""
        cases = [
            TestCase(
                id="pen_001",
                query="未经消防设计审查擅自施工会受到什么处罚？",
                expected_keywords=["停止施工", "三万元以上三十万元以下", "罚款"],
                expected_articles=["第五十八条"],
                category="建设违法",
                difficulty="hard",
                query_type="处罚",
                description="测试对建设工程违法行为处罚的理解"
            ),
            TestCase(
                id="pen_002",
                query="谎报火警会面临什么法律后果？",
                expected_keywords=["治安管理处罚法", "处罚"],
                expected_articles=["第六十二条"],
                category="违法行为",
                difficulty="medium",
                query_type="处罚",
                description="测试对谎报火警处罚的理解"
            ),
            TestCase(
                id="pen_003",
                query="过失引起火灾的法律责任是什么？",
                expected_keywords=["十日以上十五日以下拘留", "五百元以下罚款", "警告"],
                expected_articles=["第六十四条"],
                category="火灾责任",
                difficulty="medium",
                query_type="处罚",
                description="测试对过失引起火灾责任的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_special_place_queries(self):
        """生成特殊场所类查询"""
        cases = [
            TestCase(
                id="place_001",
                query="易燃易爆危险品场所的设置要求是什么？",
                expected_keywords=["不得与居住场所", "同一建筑物", "安全距离", "消防技术标准"],
                expected_articles=["第十九条", "第二十二条"],
                category="危险品场所",
                difficulty="medium",
                query_type="要求",
                description="测试对危险品场所设置要求的理解"
            ),
            TestCase(
                id="place_002",
                query="大型群众性活动的消防安全要求有哪些？",
                expected_keywords=["安全许可", "灭火和应急疏散预案", "演练", "消防设施", "疏散通道"],
                expected_articles=["第二十条"],
                category="群众活动",
                difficulty="hard",
                query_type="要求",
                description="测试对大型活动消防要求的理解"
            ),
            TestCase(
                id="place_003",
                query="禁止吸烟和使用明火的场所有哪些规定？",
                expected_keywords=["禁止吸烟", "使用明火", "审批手续", "消防安全措施"],
                expected_articles=["第二十一条"],
                category="明火管理",
                difficulty="medium",
                query_type="规定",
                description="测试对明火使用规定的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_organization_queries(self):
        """生成组织机构类查询"""
        cases = [
            TestCase(
                id="org_001",
                query="国家综合性消防救援队的职责是什么？",
                expected_keywords=["火灾扑救", "重大灾害事故", "应急救援", "抢救人员生命"],
                expected_articles=["第三十六条", "第三十七条"],
                category="救援队伍",
                difficulty="medium",
                query_type="职责",
                description="测试对消防救援队职责的理解"
            ),
            TestCase(
                id="org_002",
                query="专职消防队和志愿消防队的建立要求是什么？",
                expected_keywords=["乡镇人民政府", "经济发展", "消防工作需要", "火灾扑救"],
                expected_articles=["第三十六条"],
                category="消防队伍",
                difficulty="medium",
                query_type="要求",
                description="测试对消防队伍建立要求的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_complex_queries(self):
        """生成复合查询"""
        cases = [
            TestCase(
                id="comp_001",
                query="消防产品的质量监督体系是如何构建的？",
                expected_keywords=["国家标准", "强制性产品认证", "技术鉴定", "质量监督部门", "工商行政管理"],
                expected_articles=["第二十四条", "第二十五条"],
                category="质量监督",
                difficulty="hard",
                query_type="体系",
                description="测试对消防产品质量监督体系的综合理解"
            ),
            TestCase(
                id="comp_002",
                query="建筑消防设施的全生命周期管理包括哪些环节？",
                expected_keywords=["配置", "设置", "检验", "维修", "检测", "完好有效"],
                expected_articles=["第十六条"],
                category="设施管理",
                difficulty="hard",
                query_type="管理",
                description="测试对消防设施全生命周期管理的理解"
            ),
            TestCase(
                id="comp_003",
                query="消防宣传教育的责任主体和实施方式有哪些？",
                expected_keywords=["人民政府", "应急管理部门", "教育部门", "工会", "共青团", "妇联", "村委会"],
                expected_articles=["第六条"],
                category="宣传教育",
                difficulty="hard",
                query_type="体系",
                description="测试对消防宣传教育体系的综合理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_edge_case_queries(self):
        """生成边界情况查询"""
        cases = [
            TestCase(
                id="edge_001",
                query="森林和草原的消防工作适用消防法吗？",
                expected_keywords=["森林", "草原", "消防工作", "另有规定", "从其规定"],
                expected_articles=["第四条"],
                category="适用范围",
                difficulty="hard",
                query_type="适用性",
                description="测试对法律适用范围边界的理解"
            ),
            TestCase(
                id="edge_002",
                query="军事设施的消防工作由谁负责？",
                expected_keywords=["军事设施", "主管单位", "监督管理"],
                expected_articles=["第四条"],
                category="特殊管辖",
                difficulty="medium",
                query_type="管辖",
                description="测试对特殊领域管辖权的理解"
            ),
            TestCase(
                id="edge_003",
                query="消防法什么时候开始施行？",
                expected_keywords=["2009年5月1日", "施行"],
                expected_articles=["第七十四条"],
                category="法律效力",
                difficulty="easy",
                query_type="时效",
                description="测试对法律生效时间的理解"
            )
        ]
        self.test_cases.extend(cases)
        
        # 新增更多测试用例
        self._generate_fire_prevention_queries()
        self._generate_fire_fighting_queries()
        self._generate_supervision_queries()
        self._generate_legal_liability_queries()
        self._generate_technical_standard_queries()
        self._generate_emergency_response_queries()
        self._generate_training_education_queries()
        self._generate_equipment_facility_queries()
    
    def _generate_fire_prevention_queries(self):
        """生成火灾预防相关查询"""
        cases = [
            TestCase(
                id="prev_001",
                query="消防规划应该包括哪些内容？",
                expected_keywords=["消防安全布局", "消防站", "消防供水", "消防通信", "消防车通道", "消防装备"],
                expected_articles=["第八条"],
                category="消防规划",
                difficulty="medium",
                query_type="内容",
                description="测试对消防规划内容的理解"
            ),
            TestCase(
                id="prev_002",
                query="建设工程消防设计必须符合什么标准？",
                expected_keywords=["国家工程建设消防技术标准", "建设", "设计", "施工", "工程监理"],
                expected_articles=["第九条"],
                category="建设标准",
                difficulty="easy",
                query_type="标准",
                description="测试对建设工程消防设计标准的理解"
            ),
            TestCase(
                id="prev_003",
                query="特殊建设工程的消防设计审查由谁负责？",
                expected_keywords=["住房和城乡建设主管部门", "消防设计文件", "审查", "特殊建设工程"],
                expected_articles=["第十一条"],
                category="审查制度",
                difficulty="medium",
                query_type="程序",
                description="测试对特殊建设工程审查程序的理解"
            ),
            TestCase(
                id="prev_004",
                query="单位应当履行哪些消防安全职责？",
                expected_keywords=["消防安全责任制", "消防安全制度", "灭火和应急疏散预案", "消防设施", "防火检查"],
                expected_articles=["第十六条"],
                category="单位职责",
                difficulty="hard",
                query_type="职责",
                description="测试对单位消防安全职责的全面理解"
            ),
            TestCase(
                id="prev_005",
                query="消防安全重点单位除了一般职责外还要履行什么职责？",
                expected_keywords=["消防安全管理人", "消防档案", "每日防火巡查", "岗前消防安全培训"],
                expected_articles=["第十七条"],
                category="重点单位",
                difficulty="hard",
                query_type="职责",
                description="测试对消防安全重点单位特殊职责的理解"
            ),
            TestCase(
                id="prev_006",
                query="建筑材料和室内装修材料的防火性能要求是什么？",
                expected_keywords=["建筑构件", "建筑材料", "室内装修", "装饰材料", "国家标准", "行业标准", "不燃", "难燃材料"],
                expected_articles=["第二十六条"],
                category="材料标准",
                difficulty="medium",
                query_type="要求",
                description="测试对建筑材料防火性能要求的理解"
            ),
            TestCase(
                id="prev_007",
                query="电器产品和燃气用具的消防安全要求有哪些？",
                expected_keywords=["电器产品", "燃气用具", "产品标准", "安装", "使用", "线路", "管路", "维护保养", "检测"],
                expected_articles=["第二十七条"],
                category="产品安全",
                difficulty="medium",
                query_type="要求",
                description="测试对电器燃气产品消防要求的理解"
            ),
            TestCase(
                id="prev_008",
                query="哪些行为是被禁止的消防违法行为？",
                expected_keywords=["损坏", "挪用", "拆除", "停用", "消防设施", "埋压", "圈占", "遮挡", "消火栓", "占用", "堵塞", "封闭", "疏散通道"],
                expected_articles=["第二十八条"],
                category="禁止行为",
                difficulty="medium",
                query_type="禁止",
                description="测试对消防违法行为的识别"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_fire_fighting_queries(self):
        """生成灭火救援相关查询"""
        cases = [
            TestCase(
                id="fight_001",
                query="发现火灾时应该怎么办？",
                expected_keywords=["立即报警", "无偿", "便利", "不得阻拦", "严禁谎报火警"],
                expected_articles=["第四十四条"],
                category="火灾报警",
                difficulty="easy",
                query_type="程序",
                description="测试对火灾报警程序的理解"
            ),
            TestCase(
                id="fight_002",
                query="人员密集场所发生火灾时现场工作人员应该做什么？",
                expected_keywords=["现场工作人员", "立即组织", "引导", "在场人员疏散"],
                expected_articles=["第四十四条"],
                category="应急疏散",
                difficulty="medium",
                query_type="程序",
                description="测试对人员密集场所应急疏散的理解"
            ),
            TestCase(
                id="fight_003",
                query="火灾现场总指挥有哪些权力？",
                expected_keywords=["使用各种水源", "截断电力", "划定警戒区", "交通管制", "利用临近建筑物", "拆除", "破损", "调动供水"],
                expected_articles=["第四十五条"],
                category="现场指挥",
                difficulty="hard",
                query_type="权力",
                description="测试对火灾现场指挥权力的理解"
            ),
            TestCase(
                id="fight_004",
                query="消防车执行任务时享有哪些特殊权利？",
                expected_keywords=["不受行驶速度", "行驶路线", "行驶方向", "指挥信号", "让行", "免收车辆通行费"],
                expected_articles=["第四十七条"],
                category="特殊权利",
                difficulty="medium",
                query_type="权利",
                description="测试对消防车特殊权利的理解"
            ),
            TestCase(
                id="fight_005",
                query="消防队扑救火灾是否收费？",
                expected_keywords=["不得收取任何费用", "国家综合性消防救援队", "专职消防队"],
                expected_articles=["第四十九条"],
                category="收费政策",
                difficulty="easy",
                query_type="政策",
                description="测试对消防救援收费政策的理解"
            ),
            TestCase(
                id="fight_006",
                query="火灾事故调查由谁负责？",
                expected_keywords=["消防救援机构", "封闭火灾现场", "调查火灾原因", "统计火灾损失", "火灾事故认定书"],
                expected_articles=["第五十一条"],
                category="事故调查",
                difficulty="medium",
                query_type="程序",
                description="测试对火灾事故调查程序的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_supervision_queries(self):
        """生成监督检查相关查询"""
        cases = [
            TestCase(
                id="super_001",
                query="地方政府在消防监督检查中的职责是什么？",
                expected_keywords=["消防工作责任制", "监督检查", "有关部门", "消防安全职责"],
                expected_articles=["第五十二条"],
                category="政府监督",
                difficulty="medium",
                query_type="职责",
                description="测试对政府消防监督职责的理解"
            ),
            TestCase(
                id="super_002",
                query="消防救援机构发现火灾隐患时应该怎么处理？",
                expected_keywords=["通知", "立即采取措施", "消除隐患", "严重威胁公共安全", "临时查封措施"],
                expected_articles=["第五十四条"],
                category="隐患处理",
                difficulty="medium",
                query_type="程序",
                description="测试对火灾隐患处理程序的理解"
            ),
            TestCase(
                id="super_003",
                query="消防监督检查应该遵循什么原则？",
                expected_keywords=["公正", "严格", "文明", "高效", "不得收取费用", "不得利用职务谋取利益"],
                expected_articles=["第五十六条"],
                category="检查原则",
                difficulty="medium",
                query_type="原则",
                description="测试对消防监督检查原则的理解"
            ),
            TestCase(
                id="super_004",
                query="公民对消防执法有什么监督权利？",
                expected_keywords=["检举", "控告", "违法行为", "及时查处"],
                expected_articles=["第五十七条"],
                category="公民监督",
                difficulty="easy",
                query_type="权利",
                description="测试对公民消防监督权利的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_legal_liability_queries(self):
        """生成法律责任相关查询"""
        cases = [
            TestCase(
                id="legal_001",
                query="建设单位违反消防设计审查规定会面临什么处罚？",
                expected_keywords=["停止施工", "停止使用", "停产停业", "三万元以上三十万元以下", "罚款"],
                expected_articles=["第五十八条"],
                category="建设违法处罚",
                difficulty="hard",
                query_type="处罚",
                description="测试对建设工程违法处罚的理解"
            ),
            TestCase(
                id="legal_002",
                query="单位违反消防设施管理规定的处罚标准是什么？",
                expected_keywords=["五千元以上五万元以下", "罚款", "消防设施", "器材", "消防安全标志"],
                expected_articles=["第六十条"],
                category="设施违法处罚",
                difficulty="medium",
                query_type="处罚",
                description="测试对消防设施违法处罚的理解"
            ),
            TestCase(
                id="legal_003",
                query="个人违反消防安全规定会受到什么处罚？",
                expected_keywords=["警告", "五百元以下罚款", "五日以下拘留", "十日以上十五日以下拘留"],
                expected_articles=["第六十条", "第六十三条", "第六十四条"],
                category="个人违法处罚",
                difficulty="medium",
                query_type="处罚",
                description="测试对个人消防违法处罚的理解"
            ),
            TestCase(
                id="legal_004",
                query="消防技术服务机构违法行为的处罚措施有哪些？",
                expected_keywords=["五万元以上十万元以下", "罚款", "停止执业", "吊销相应资格", "终身市场禁入"],
                expected_articles=["第六十九条"],
                category="技术服务违法",
                difficulty="hard",
                query_type="处罚",
                description="测试对消防技术服务违法处罚的理解"
            ),
            TestCase(
                id="legal_005",
                query="消防工作人员滥用职权会承担什么责任？",
                expected_keywords=["滥用职权", "玩忽职守", "徇私舞弊", "依法给予处分", "构成犯罪", "追究刑事责任"],
                expected_articles=["第七十一条", "第七十二条"],
                category="职务违法责任",
                difficulty="hard",
                query_type="责任",
                description="测试对消防工作人员违法责任的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_technical_standard_queries(self):
        """生成技术标准相关查询"""
        cases = [
            TestCase(
                id="tech_001",
                query="消防产品必须符合什么标准？",
                expected_keywords=["国家标准", "行业标准", "强制性产品认证", "技术鉴定"],
                expected_articles=["第二十四条"],
                category="产品标准",
                difficulty="medium",
                query_type="标准",
                description="测试对消防产品标准要求的理解"
            ),
            TestCase(
                id="tech_002",
                query="建设工程消防设计审查验收的具体办法由谁制定？",
                expected_keywords=["国务院住房和城乡建设主管部门", "具体办法"],
                expected_articles=["第十四条"],
                category="制度规定",
                difficulty="medium",
                query_type="规定",
                description="测试对消防制度制定权限的理解"
            ),
            TestCase(
                id="tech_003",
                query="消防安全检查的具体办法由哪个部门制定？",
                expected_keywords=["国务院应急管理部门", "消防安全检查", "具体办法"],
                expected_articles=["第十五条"],
                category="检查制度",
                difficulty="medium",
                query_type="规定",
                description="测试对消防检查制度制定权限的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_emergency_response_queries(self):
        """生成应急响应相关查询"""
        cases = [
            TestCase(
                id="emerg_001",
                query="地方政府应该如何建立火灾应急响应机制？",
                expected_keywords=["应急预案", "应急反应", "处置机制", "人员", "装备", "保障"],
                expected_articles=["第四十三条"],
                category="应急机制",
                difficulty="hard",
                query_type="机制",
                description="测试对火灾应急响应机制的理解"
            ),
            TestCase(
                id="emerg_002",
                query="参加扑救火灾受伤的人员享受什么待遇？",
                expected_keywords=["医疗", "抚恤", "受伤", "致残", "死亡", "国家有关规定"],
                expected_articles=["第五十条"],
                category="伤亡抚恤",
                difficulty="medium",
                query_type="待遇",
                description="测试对消防伤亡抚恤政策的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_training_education_queries(self):
        """生成培训教育相关查询"""
        cases = [
            TestCase(
                id="train_001",
                query="各级政府在消防宣传教育中的职责是什么？",
                expected_keywords=["经常性", "消防宣传教育", "提高", "消防安全意识"],
                expected_articles=["第六条"],
                category="宣传教育",
                difficulty="medium",
                query_type="职责",
                description="测试对政府消防宣传教育职责的理解"
            ),
            TestCase(
                id="train_002",
                query="哪些人员必须持证上岗？",
                expected_keywords=["电焊", "气焊", "火灾危险作业", "自动消防系统", "操作人员", "持证上岗"],
                expected_articles=["第二十一条"],
                category="持证上岗",
                difficulty="medium",
                query_type="要求",
                description="测试对特殊作业人员持证要求的理解"
            ),
            TestCase(
                id="train_003",
                query="教育部门在消防教育中承担什么责任？",
                expected_keywords=["教育", "人力资源行政主管部门", "学校", "职业培训机构", "消防知识", "教育", "教学", "培训"],
                expected_articles=["第六条"],
                category="教育培训",
                difficulty="medium",
                query_type="责任",
                description="测试对教育部门消防教育责任的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def _generate_equipment_facility_queries(self):
        """生成设备设施相关查询"""
        cases = [
            TestCase(
                id="equip_001",
                query="公共消防设施包括哪些内容？",
                expected_keywords=["消防供水", "消防通信", "消防车通道", "完好有效"],
                expected_articles=["第二十九条"],
                category="公共设施",
                difficulty="medium",
                query_type="内容",
                description="测试对公共消防设施内容的理解"
            ),
            TestCase(
                id="equip_002",
                query="哪些单位应当建立单位专职消防队？",
                expected_keywords=["大型核设施", "大型发电厂", "民用机场", "主要港口", "易燃易爆危险品", "重要物资", "古建筑群"],
                expected_articles=["第三十九条"],
                category="专职消防队",
                difficulty="hard",
                query_type="要求",
                description="测试对专职消防队建立要求的理解"
            ),
            TestCase(
                id="equip_003",
                query="消防车和消防器材能否用于其他用途？",
                expected_keywords=["不得用于", "消防和应急救援工作无关", "事项"],
                expected_articles=["第四十八条"],
                category="设备使用",
                difficulty="easy",
                query_type="规定",
                description="测试对消防设备专用性规定的理解"
            ),
            TestCase(
                id="equip_004",
                query="建筑消防设施检测的要求是什么？",
                expected_keywords=["每年至少", "一次全面检测", "完好有效", "检测记录", "完整准确", "存档备查"],
                expected_articles=["第十六条"],
                category="设施检测",
                difficulty="medium",
                query_type="要求",
                description="测试对建筑消防设施检测要求的理解"
            )
        ]
        self.test_cases.extend(cases)
    
    def get_test_cases_by_category(self, category: str) -> List[TestCase]:
        """按类别获取测试用例"""
        return [case for case in self.test_cases if case.category == category]
    
    def get_test_cases_by_difficulty(self, difficulty: str) -> List[TestCase]:
        """按难度获取测试用例"""
        return [case for case in self.test_cases if case.difficulty == difficulty]
    
    def get_test_cases_by_type(self, query_type: str) -> List[TestCase]:
        """按查询类型获取测试用例"""
        return [case for case in self.test_cases if case.query_type == query_type]
    
    def get_random_test_cases(self, count: int) -> List[TestCase]:
        """随机获取测试用例"""
        return random.sample(self.test_cases, min(count, len(self.test_cases)))
    
    def export_to_json(self, output_file: str):
        """导出测试用例到JSON文件"""
        data = {
            "metadata": {
                "total_cases": len(self.test_cases),
                "categories": list(set(case.category for case in self.test_cases)),
                "difficulties": list(set(case.difficulty for case in self.test_cases)),
                "query_types": list(set(case.query_type for case in self.test_cases))
            },
            "test_cases": [asdict(case) for case in self.test_cases]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"测试用例已导出到: {output_file}")
        print(f"总计: {len(self.test_cases)} 个测试用例")
    
    def print_statistics(self):
        """打印测试用例统计信息"""
        print("测试用例统计信息:")
        print(f"总计: {len(self.test_cases)} 个测试用例")
        
        # 按类别统计
        categories = {}
        for case in self.test_cases:
            categories[case.category] = categories.get(case.category, 0) + 1
        
        print("\n按类别分布:")
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count}")
        
        # 按难度统计
        difficulties = {}
        for case in self.test_cases:
            difficulties[case.difficulty] = difficulties.get(case.difficulty, 0) + 1
        
        print("\n按难度分布:")
        for difficulty, count in sorted(difficulties.items()):
            print(f"  {difficulty}: {count}")
        
        # 按查询类型统计
        query_types = {}
        for case in self.test_cases:
            query_types[case.query_type] = query_types.get(case.query_type, 0) + 1
        
        print("\n按查询类型分布:")
        for query_type, count in sorted(query_types.items()):
            print(f"  {query_type}: {count}")

def main():
    """主函数"""
    generator = TestDataGenerator()
    
    # 打印统计信息
    generator.print_statistics()
    
    # 导出测试用例
    output_file = "/home/ubuntu/workspace/know_ledgebase/tests/test_cases.json"
    generator.export_to_json(output_file)
    
    # 示例：获取不同类型的测试用例
    print("\n示例查询:")
    easy_cases = generator.get_test_cases_by_difficulty("easy")
    print(f"简单查询示例: {easy_cases[0].query}")
    
    hard_cases = generator.get_test_cases_by_difficulty("hard")
    print(f"困难查询示例: {hard_cases[0].query}")
    
    random_cases = generator.get_random_test_cases(3)
    print("\n随机查询示例:")
    for case in random_cases:
        print(f"  {case.query} ({case.difficulty})")

if __name__ == "__main__":
    main()