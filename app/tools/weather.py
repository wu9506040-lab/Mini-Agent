"""weather 工具（mock 实现）：确定性返回，预留真实 API 接入位。"""
from typing import Optional

from pydantic import BaseModel, Field

from .base import BaseTool, register_tool

_MOCK_WEATHER = {
    "北京": ("晴", 25),
    "上海": ("多云", 28),
    "广州": ("雷阵雨", 31),
    "深圳": ("多云转晴", 30),
    "东莞": ("晴间多云", 29),
}


class WeatherArgs(BaseModel):
    city: str = Field(..., description="城市名，例如 北京")
    date: Optional[str] = Field(None, description="日期（可选），例如 今天 / 明天 / 2026-11-10")


@register_tool
class WeatherTool(BaseTool):
    name = "weather"
    description = "查询指定城市某天的天气。用户问天气、出行要不要带伞时调用。"
    args_model = WeatherArgs

    async def run(self, city: str, date: Optional[str] = None) -> str:
        # mock：确定性数据。真实接入点：调用天气 API（加超时与降级）。
        day = date or "今天"
        if city in _MOCK_WEATHER:
            condition, temp = _MOCK_WEATHER[city]
            return f"[mock 天气] {city} {day}：{condition}，{temp}°C"
        return f"[mock 天气] {city} {day}：晴，25°C（mock 默认值，该城市无记录）"
