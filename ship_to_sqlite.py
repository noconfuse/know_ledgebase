import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect, MetaData
from pathlib import Path
from typing import List, Dict, Optional
from app.config import modelsettings

class ExcelToSqlite:
    def __init__(self, 
                 file_path: str,
                 db_path: str,
                 table_name: str,
                 column_mapping: Dict[str, str],
                 table_description: str = None,
                 column_descriptions: Dict[str, str] = None):
        """
        初始化转换器
        :param file_path: Excel文件路径
        :param db_path: SQLite数据库文件路径
        :param table_name: 数据表名
        :param column_mapping: 列名映射字典
        :param table_description: 表格描述
        :param column_descriptions: 字段描述字典
        """

        # 检查文件是否存在，没有则创建
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        self.file_path = file_path
        self.db_path = f"sqlite:///{db_path}"
        self.table_name = table_name
        self.column_mapping = column_mapping
        self.table_description = table_description
        self.column_descriptions = column_descriptions or {}
        
        # 初始化数据库连接和元数据
        self.engine = create_engine(self.db_path)
        self.metadata = MetaData()

    def read_excel_file(self, file_path: Path) -> List[Dict]:
        """读取Excel文件并转换列名"""
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # 清理列名，移除冒号
        df.columns = df.columns.str.rstrip(':')
        
        # 确保所有列名都在映射中
        if not all(col in self.column_mapping for col in df.columns):
            missing_cols = [col for col in df.columns if col not in self.column_mapping]
            raise ValueError(f"以下列名在column_mapping中未找到映射: {missing_cols}")
        
        df.rename(columns=self.column_mapping, inplace=True)
        return df.to_dict("records")

    def get_sql_type(self, dtype):
        """自动推断SQLite数据类型"""
        if pd.api.types.is_integer_dtype(dtype):
            return 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype):
            return 'REAL'
        else:
            return 'TEXT'

    def create_table(self, df):
        """动态创建数据表，添加表格和字段描述"""
        from sqlalchemy import Table, Column, String, Integer, Float, DateTime, MetaData
        
        # 创建表对象
        table = Table(
            self.table_name,
            self.metadata,
            extend_existing=True,
            comment=self.table_description,  # 添加表描述
        )
        
        # 添加元数据字段
        table.append_column(Column('_source_file', String, comment='源文件名'))
        table.append_column(Column('_import_time', DateTime, comment='数据导入时间'))
        table.append_column(Column('_data_version', String, comment='数据版本'))
        table.append_column(Column('_data_type', String, comment='数据类型'))
        
        # 添加数据字段
        for col, dtype in df.dtypes.items():
            sql_type = self.get_sql_type(dtype)
            column = Column(
                col,
                {
                    'INTEGER': Integer,
                    'REAL': Float,
                    'TEXT': String
                }[sql_type],
                comment=self.column_descriptions.get(col, '')  # 添加字段描述
            )
            table.append_column(column)
        
        # 创建表
        self.metadata.create_all(self.engine)
        return table

def main():
    PROJECT_NAME = "ship_check"
    base_dir = f"{modelsettings.PROJECTS_DIR}/{PROJECT_NAME}/documents/ship_info"
    db_path = f"{base_dir}/database/ships.db"

    directory_configs = {
        f"{base_dir}/国际船舶录.xlsx": {
            "table_name": "internal_ships",
            "column_mapping": {
                "船舶登记号/船检号": "ship_registration_number",
                "船名(中文)": "ship_name_chinese",
                "船名(英文)": "ship_name_english",
                "船舶呼号": "ship_call_sign",
                "IMO No.": "imo_number",
                "曾用名": "former_names",
                "船旗国": "flag_country",
                "船籍港": "port_of_registration",
                "船东": "shipowner",
                "船舶经营人": "ship_operator",
                "船舶类型及用途": "ship_type_and_usage",
                "下次特检日期": "next_inspection_date",
                "总吨位": "gross_tonnage",
                "净吨位": "net_tonnage",
                "载重吨": "payload_tonnage",
                "船舶总长": "ship_length",
                "垂线间长": "vertical_line_length",
                "型宽": "breadth",
                "型深": "depth",
                "干舷": "draught",
                "平均吃水": "average_discharge",
                "船速": "speed",
                "船体入级符号": "hull_grading_symbol",
                "船体附加标志": "hull_additional_indicators",
                "轮机入级符号": "main_engine_grading_symbol",
                "轮机附加标志": "main_engine_additional_indicators",
                "上层建筑及甲板室": "upper_buildings_and_deck_rooms",
                "甲板层数": "number_of_deck_layers",
                "舱壁数": "number_of_cabin_walls",
                "压载": "payload",
                "船舶建造厂": "ship_building_factory",
                "船舶建造地点": "ship_building_location",
                "船舶建造时间": "ship_building_time",
                "重大改建部位的建造日期": "date_of_construction_of_major_reform_parts",
                "重大改建情况": "major_reform_case",
                "主机建造厂": "main_engine_factory",
                "锅炉建造厂": "boiler_factory",
                "主机建造时间": "main_engine_building_time",
                "锅炉建造时间": "boiler_building_time",
                "主机型式*数量": "main_engine_type_and_quantity",
                "主机缸数*缸径*功率*转速": "main_engine_cylinder_number_cylinder_diameter_power_rpm",
                "发电机*功率*电压*数": "generator_power_voltage_number",
                "锅炉数*压力*受热面积": "boiler_number_pressure_heat_area",
                "集装箱数*规格": "container_number_specification",
                "货舱数": "cargo_hold_number",
                "货舱总容积": "total_cargo_hold_volume",
                "舱口数及舱口尺度": "number_of_cabin_ports_and_cabin_port_scale",
                "起货设备类型,安全负荷,数量": "loading_device_type_safety_load_quantity",
                "各货舱容积": "cargo_hold_volume"
            }
        },
        f"{base_dir}/国内船舶录.xlsx": {
            "table_name": "domestic_ships",
            "column_mapping": {
                # 船名	船检登记号	呼号	CCS NO.	IMO NO.	海河船	机动非机动	船舶归属类别	入级状态	曾用名	船级港	船舶类型	航区	船舶所有人	船舶经营人	船舶制造厂	船舶入级符合	船体附加标志	轮机入级符号	轮机附加标志	总吨	净吨	载重吨	夏季干舷	吃水(满载)	航速	船舶总长	垂线间长	型宽	型深	甲板层数	水密横舱壁数	货舱数	船舶完工日期	改建日期	载运标准集装箱数量（货舱内）	载运标准集装箱数量（货舱舱口盖上）	主机_数量	主机_型式	主机_制造厂	主机_制造日期	主机_缸数	主机_缸径	主机_行程	主机_额定功率	主机_额定转速	锅炉_数量	锅炉_设计压力(Mpa)	锅炉_受热面积(m2)	锅炉_制造厂	核定干舷（A级）	核定干舷（B级）	核定干舷（C级）	核定干舷（J1）	核定干舷（J2）
                "船名": "ship_name",
                "船检登记号": "ship_inspection_registration_number",
                "呼号": "call_sign",
                "CCS NO.": "CCS_NO",
                "IMO NO.": "IMO_NO",
                "海河船":  "sea_or_river_ship", #海船或河船
                "机动非机动": "amateur_or_professional",
                "船舶归属类别": "ship_affiliation_category",
                "入级状态": "grading_status",
                "曾用名": "former_names",
                "船级港": "ship_grading_port",
                "船舶类型": "ship_type",
                "航区": "area",
                "船舶所有人": "ship_owner",
                "船舶经营人": "ship_operator",
                "船舶制造厂": "ship_factory",
                "船舶入级符合": "ship_grading_conform",
                "船体附加标志": "hull_additional_indicators",
                "轮机入级符号": "main_engine_grading_symbol",
                "轮机附加标志": "main_engine_additional_indicators",
                "总吨": "gross_tonnage",
                "净吨": "net_tonnage",
                "载重吨": "payload_tonnage",
                "夏季干舷": "summer_draught",
                "吃水(满载)": "average_discharge",
                "航速": "speed",
                "船舶总长": "ship_length",
                "垂线间长": "vertical_line_length",
                "型宽": "breadth",
                "型深": "depth",
                "甲板层数": "number_of_deck_layers",
                "水密横舱壁数": "water_dense_cross_cabin_walls",
                "货舱数": "cargo_hold_number",
                "船舶完工日期": "ship_completion_date",
                "改建日期": "remodelling_date",
                "载运标准集装箱数量（货舱内）": "loading_standard_container_quantity_cargo_hold",
                "载运标准集装箱数量（货舱舱口盖上）": "loading_standard_container_quantity_cargo_hold_port_cover",
                "主机_数量": "main_engine_number",
                "主机_型式": "main_engine_type",
                "主机_制造厂": "main_engine_factory",
                "主机_制造日期": "main_engine_manufacturing_date",
                "主机_缸数": "main_engine_cylinder_number",
                "主机_缸径": "main_engine_cylinder_diameter",
                "主机_行程": "main_engine_rpm",
                "主机_额定功率": "main_engine_rated_power",
                "主机_额定转速": "main_engine_rated_rpm",
                "锅炉_数量": "boiler_number",
                "锅炉_设计压力(Mpa)": "boiler_design_pressure_Mpa",
                "锅炉_受热面积(m2)": "boiler_heat_area_m2",
                "锅炉_制造厂": "boiler_factory",
                "核定干舷（A级）": "approved_draught_A_level",
                "核定干舷（B级）": "approved_draught_B_level",
                "核定干舷（C级）": "approved_draught_C_level",
                "核定干舷（J1）": "approved_draught_J1",
                "核定干舷（J2）": "approved_draught_J2"
            }
        }
    }
    
    # 处理每个文件
    for file_path, config in directory_configs.items():
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        print(f"\n处理文件: {file_path}")
        print(f"使用表名: {config['table_name']}")
        
        try:
            converter = ExcelToSqlite(
                file_path=file_path,
                db_path=db_path,
                table_name=config['table_name'],
                column_mapping=config['column_mapping'],
                table_description=config.get('table_description', ''),
                column_descriptions=config.get('column_descriptions', {})
            )
        
            converter.process()
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

if __name__ == "__main__":
    main()





