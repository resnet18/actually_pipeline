@echo off
chcp 65001 >nul
cd /d D:\Projects\actually_pipeline

echo.
echo ========================================
echo  1/4 普通版预训练 (CC 通用语料)
echo ========================================
python src\train.py
if errorlevel 1 goto error

echo.
echo ========================================
echo  2/4 MoE版预训练 (CC 通用语料)
echo ========================================
python src\train_moe.py
if errorlevel 1 goto error

echo.
echo ========================================
echo  3/4 普通版继续预训练 (ACL 领域语料)
echo ========================================
python src\train_cpt.py
if errorlevel 1 goto error

echo.
echo ========================================
echo  4/4 MoE版继续预训练 (ACL 领域语料)
echo ========================================
python src\train_moe_cpt.py
if errorlevel 1 goto error

echo.
echo ========================================
echo  全部完成！可以睡觉了（笑）
echo ========================================
goto end

:error
echo.
echo ========================================
echo  训练中断，请检查上面的错误输出
echo ========================================

:end
pause