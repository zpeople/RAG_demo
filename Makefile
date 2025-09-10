# 安装开发依赖
install-dev:
	pip install -r requirements-dev.txt


# 运行所有代码质量检查
quality:
	black --check .               # 检查代码格式
	isort --check .               # 检查导入排序
	flake8 .                      # 代码风格检查
	mypy .                        # 类型检查
	bandit -r .                   # 安全扫描
	safety check                  # 依赖安全检查
	radon cc . -a                 # 复杂度分析 (cc: cyclomatic complexity)
	radon mi . -s                 # 可维护性指数分析

# 自动修复可修复的问题
fix:
	black .                       # 自动格式化代码
	isort .                       # 自动排序导入

# 运行测试并生成覆盖率报告
test:
	pytest tests/ --cov=src/ --cov-report=term --cov-report=html
	@echo "覆盖率报告已生成: htmlcov/index.html"

# CI流程：运行所有检查和测试
ci: quality test
	