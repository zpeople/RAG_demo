import inspect  # 用于判断函数是否为异步函数

def skip_execution(skip=True):
    def decorator(func):
        # 判断被装饰函数是否为异步函数（async def 定义）
        if inspect.iscoroutinefunction(func):
            # 异步函数的 wrapper（必须用 async def 定义）
            async def async_wrapper(*args, **kwargs):
                if not skip:
                    # 异步函数需用 await 调用
                    return await func(*args, **kwargs)
                print(f"异步函数 {func.__name__} 已跳过执行")
                return None
            return async_wrapper
        else:
            # 同步函数的 wrapper（原逻辑不变）
            def sync_wrapper(*args, **kwargs):
                if not skip:
                    return func(*args, **kwargs)
                print(f"同步函数 {func.__name__} 已跳过执行")
                return None
            return sync_wrapper
    return decorator

def train_execution(train=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if train:
                return func(*args, **kwargs)
            # 如果skip为True，则不执行函数，返回None或提示
            print(f"跳过训练 {func.__name__} 已跳过执行")
            return None

        return wrapper

    return decorator
